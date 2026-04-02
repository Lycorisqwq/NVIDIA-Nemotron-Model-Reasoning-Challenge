#!/usr/bin/env python3
"""
NVIDIA Nemotron Challenge - Pure BF16 LoRA via Unsloth
Tailored for A100 80G - Maximize Reasoning Accuracy
"""

import argparse
import os
import gc
import sys
import torch
import zipfile
import polars as pl
import pandas as pd
from datasets import Dataset

# ============================================================
# 0. 阻断网络请求，强制离线模式 (解决 120s Timeout 报错)
# ============================================================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["DISABLE_TELEMETRY"] = "1"

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

parser = argparse.ArgumentParser(description="Fine-tune Nemotron-30B with Pure BF16 LoRA")
parser.add_argument("--model_path", type=str,
                    default="/home/NVIDIA-Nemotron-Model-Reasoning-Challenge/model/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1")
parser.add_argument("--data_path", type=str,
                    default="/home/NVIDIA-Nemotron-Model-Reasoning-Challenge/data/train_full.csv")
parser.add_argument("--output_dir", type=str, default="output/adapter")
parser.add_argument("--zip_path", type=str, default="output/submission.zip")
args = parser.parse_args()


# ============================================================
#  RMSNorm Fix (pure PyTorch fallback for stability)
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
# 1. 超参数配置 (纯 BF16 比较吃显存，参数略微保守)
# ============================================================
MAX_SEQ_LEN = 2048     
LORA_RANK = 32         # 纯 BF16 下，Rank 32 是一个非常稳妥且高效的值
NUM_EPOCHS = 2
BATCH_SIZE = 8         
GRAD_ACCUM = 4         # 增加梯度累积，等效 Batch Size = 8
LR = 1e-5              # 纯 BF16 的学习率通常比 QLoRA (2e-4) 要设置得低一点，防止过拟合

os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(os.path.dirname(args.zip_path) if os.path.dirname(args.zip_path) else ".", exist_ok=True)

# ============================================================
# 2. 数据处理 (保持不变)
# ============================================================
print(f"Loading training data from {args.data_path} ...")
train_df = pl.read_csv(args.data_path)

# 过滤掉 error 列中有内容的行，剔除包含 HTTP 502 等报错的数据
train_df = train_df.filter(pl.col("is_valid") == 1)

hf_dataset = Dataset.from_pandas(train_df.to_pandas())

from transformers import AutoTokenizer
# local_files_only=True 确保绝对不会去连网
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def build_training_text(example):
    prompt = example["prompt"]
    answer = example["original_answer"]
    cot = example["generated_cot"]
    user_msg = prompt + "\nPut your final answer inside \\boxed{}."
    assistant_msg = f"{cot}\n\n\\boxed{{{answer}}}"

    try:
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        text = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>"
    return {"text": text}

hf_dataset = hf_dataset.map(build_training_text, remove_columns=hf_dataset.column_names)
print(f"Dataset ready: {len(hf_dataset)} examples")

# ============================================================
# 3. 加载纯 BF16 模型
# ============================================================
print(f"\nLoading model in PURE BF16 from {args.model_path} ...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = args.model_path,
    max_seq_length = MAX_SEQ_LEN,
    dtype = torch.bfloat16,    # 强制 BF16
    load_in_4bit = False,      # 关闭 QLoRA，开启纯血 LoRA
    load_in_8bit = False,
    unsloth_force_compile = False,
    trust_remote_code = True,
    local_files_only = True,   # 强制离线加载
)

model = FastLanguageModel.get_peft_model(
    model,
    r = LORA_RANK,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"], 
    lora_alpha = LORA_RANK * 2,
    lora_dropout = 0, 
    bias = "none",
    use_gradient_checkpointing = "unsloth", # 救命稻草：保住最后 20G 显存的关键
    random_state = 3407,
)
model.gradient_checkpointing_enable()
model.print_trainable_parameters()


# ============================================================
# 3.5 定义 Loss 记录回调函数 (新增)
# ============================================================
from transformers import TrainerCallback

class LossLoggingCallback(TrainerCallback):
    def __init__(self, log_path="train_loss.log"):
        self.log_path = log_path
        # 初始化时，清空（或创建）文件并写入表头 (CSV格式，方便后续画图)
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("step, epoch, loss, learning_rate\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        # trainer 根据 logging_steps 触发打印时，会调用此方法
        if logs is not None and "loss" in logs:
            step = state.global_step
            epoch = logs.get("epoch", state.epoch)
            loss = logs["loss"]
            lr = logs.get("learning_rate", 0.0)
            
            # 实时将数据追加到日志文件中
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"{step}, {epoch:.4f}, {loss:.4f}, {lr:.6e}\n")
                
# ============================================================
# 4. 训练配置
# ============================================================
training_args = SFTConfig(
    output_dir = args.output_dir,
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACCUM,
    num_train_epochs = NUM_EPOCHS,
    learning_rate = LR, 
    logging_steps = 5,
    bf16 = True,
    max_grad_norm = 1.0,
    optim = "paged_adamw_8bit", # 模型虽然是 16-bit 的，但优化器状态我们依然用 8-bit 压缩以省显存
    lr_scheduler_type = "cosine",
    warmup_ratio = 0.1,
    save_strategy = "no",
    report_to = "none",
    dataset_text_field = "text",
    max_length = MAX_SEQ_LEN,
    packing = False,
    gradient_checkpointing = True,
    gradient_checkpointing_kwargs = {"use_reentrant": False},
    dataloader_num_workers = 4,      # 开启多线程加载数据，防止 GPU 等待 CPU 预处理文本
    dataloader_pin_memory = True,    # 开启锁页内存，加速 CPU 到 GPU 的数据传输
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = hf_dataset,
    args = training_args,
    callbacks = [LossLoggingCallback("train_loss.log")], # <--- 新增这一行
)

print("\n" + "=" * 60)
print("Starting Pure BF16 Training on A100...")
print("=" * 60)
trainer.train()

# ============================================================
# 5. 保存与打包
# ============================================================
print(f"\nSaving adapter to {args.output_dir} ...")
model.save_pretrained(args.output_dir) 
tokenizer.save_pretrained(args.output_dir)

del model, trainer
gc.collect()
torch.cuda.empty_cache()

print(f"\nPackaging files from {args.output_dir}...")
with zipfile.ZipFile(args.zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fname in os.listdir(args.output_dir):
        fpath = os.path.join(args.output_dir, fname)
        zf.write(fpath, fname)

print("\nsubmission.zip is ready! Go win the competition!")
