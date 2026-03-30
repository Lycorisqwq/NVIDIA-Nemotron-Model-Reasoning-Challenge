#!/usr/bin/env python3
"""
NVIDIA Nemotron Model Reasoning Challenge - Training Script
Adapted for A100 80G server from Kaggle notebook.

Usage:
    python train.py [--model_path MODEL_PATH] [--data_path DATA_PATH] [--output_dir OUTPUT_DIR]
"""

import argparse
import os
import sys
import stat
import shutil
import gc
import zipfile

# ============================================================
# 0. Parse arguments
# ============================================================
parser = argparse.ArgumentParser(description="Fine-tune Nemotron-3-Nano-30B with LoRA")
parser.add_argument("--model_path", type=str,
                    default="/data/hz/models/Nemotron-3-Nano-30B",
                    help="Path to the base model on disk")
parser.add_argument("--data_path", type=str,
                    default="data/final_Nemotron_training_data.csv",
                    help="Path to the training CSV")
parser.add_argument("--output_dir", type=str,
                    default="output/adapter",
                    help="Directory to save LoRA adapter")
parser.add_argument("--zip_path", type=str,
                    default="output/submission.zip",
                    help="Path for the submission zip file")
parser.add_argument("--qlora", action="store_true",
                    help="Use 4-bit QLoRA to reduce VRAM usage (~15GB vs ~60GB for model weights)")
args = parser.parse_args()

MODEL_PATH = args.model_path
DATA_PATH = args.data_path
OUTPUT_DIR = args.output_dir
ZIP_PATH = args.zip_path

# ============================================================
# 1. Environment Setup
# ============================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ============================================================
# 2. RMSNorm Fix (pure PyTorch fallback for stability)
# ============================================================
def _pure_rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-5,
                     group_size=None, norm_before_gate=True, upcast=True):
    dtype = x.dtype
    if upcast:
        x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    out = x_normed * weight.float()
    if bias is not None:
        out = out + bias.float()
    if z is not None:
        out = out * F.silu(z.float())
    return out.to(dtype)

for name, mod in list(sys.modules.items()):
    if hasattr(mod, 'rmsnorm_fn'):
        mod.rmsnorm_fn = _pure_rmsnorm_fn
        print(f"Patched rmsnorm_fn in {name}")


# ============================================================
# 3. Hyperparameters
# ============================================================
LORA_RANK = 32
MAX_SEQ_LEN = 2048
NUM_EPOCHS = 2
BATCH_SIZE = 1
GRAD_ACCUM = 4
LR = 1e-4

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(ZIP_PATH) if os.path.dirname(ZIP_PATH) else ".", exist_ok=True)

# ============================================================
# 4. Load Data
# ============================================================
print(f"\nLoading training data from {DATA_PATH} ...")
train_df = pd.read_csv(DATA_PATH)
print(f"Training samples: {len(train_df)}")
print(f"Columns: {list(train_df.columns)}")

hf_dataset = Dataset.from_pandas(train_df)

# ============================================================
# 5. Build Training Prompts
# ============================================================
print(f"\nLoading tokenizer from {MODEL_PATH} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def build_training_text(example):
    prompt = example["prompt"]
    answer = example["answer"]
    cot = example["generated_cot"]

    user_msg = prompt + "\nPut your final answer inside \\boxed{}."

    # Combine the CoT with the final answer
    assistant_msg = f"{cot}\n\n\\boxed{{{answer}}}"

    try:
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        text = (
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_msg}<|im_end|>"
        )
    return {"text": text}


print("Building training texts ...")
hf_dataset = hf_dataset.map(build_training_text, remove_columns=hf_dataset.column_names)
print(f"Dataset ready: {len(hf_dataset)} examples")

# ============================================================
# 6. Load Model & Apply LoRA
# ============================================================
print(f"\nLoading model from {MODEL_PATH} ...")
if args.qlora:
    print("Using QLoRA (4-bit quantization)")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map={"": 0},
        trust_remote_code=True,
        quantization_config=bnb_config,
    )
    model = prepare_model_for_kbit_training(model)
else:
    print("Using LoRA (bf16 full precision)")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map={"": 0},
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
model.gradient_checkpointing_enable()

# Patch: disable fast path kernels for stability
for name, mod in sys.modules.items():
    if "modeling_nemotron_h" in name:
        mod.is_fast_path_available = False
        print(f"Patched {name}: is_fast_path_available = False")

# Re-apply rmsnorm fix after model loading (new modules may have been imported)
for name, mod in list(sys.modules.items()):
    if hasattr(mod, 'rmsnorm_fn'):
        mod.rmsnorm_fn = _pure_rmsnorm_fn

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============================================================
# 7. Training
# ============================================================
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    logging_steps=5,
    bf16=True,
    max_grad_norm=1.0,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    save_strategy="no",
    report_to="none",
    dataset_text_field="text",
    max_length=MAX_SEQ_LEN,
    packing=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

trainer = SFTTrainer(
    model=model,
    train_dataset=hf_dataset,
    processing_class=tokenizer,
    args=training_args,
)

print("\n" + "=" * 60)
print("Starting training...")
print("=" * 60)
trainer.train()

# Save adapter
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nAdapter saved to {OUTPUT_DIR}:")
for f in os.listdir(OUTPUT_DIR):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    print(f"  {f} ({size / 1024:.1f} KB)")

# ============================================================
# 8. Clean up GPU memory
# ============================================================
del model, trainer
gc.collect()
torch.cuda.empty_cache()

# ============================================================
# 9. Create Submission ZIP (logic unchanged from Kaggle notebook)
# ============================================================
print(f"\nPackaging files from {OUTPUT_DIR}...")

with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fname in os.listdir(OUTPUT_DIR):
        fpath = os.path.join(OUTPUT_DIR, fname)
        zf.write(fpath, fname)

print(f"Created {ZIP_PATH} ({os.path.getsize(ZIP_PATH) / 1024 / 1024:.1f} MB)")

with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    zip_contents = zf.namelist()
    print(f"Zip Contents: {zip_contents}")

    if "adapter_config.json" not in zip_contents:
        raise AssertionError(
            "CRITICAL ERROR: adapter_config.json is missing from the zip. "
            "The Kaggle evaluation will fail."
        )
    if "adapter_model.safetensors" not in zip_contents:
        raise AssertionError(
            "CRITICAL ERROR: adapter_model.safetensors is missing from the zip."
        )

print("\nsubmission.zip is ready! You can now submit this file to the competition.")
