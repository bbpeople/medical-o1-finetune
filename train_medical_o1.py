"""
=============================================================================
 Qwen2.5-0.5B-Instruct + Unsloth + QLoRA 医疗问答微调
=============================================================================
 基座: unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit
 数据: FreedomIntelligence/medical-o1-reasoning-SFT (英文子集)
 硬件: RTX 3050 (4GB VRAM)
 方法: QLoRA (4-bit NF4) + Unsloth 优化内核
=============================================================================
"""

import os
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"       # 跳过 fused CE，走标准 logits（4GB VRAM 必须）
import torch
import argparse
from datasets import load_dataset
from datetime import datetime

# ── Unsloth ──────────────────────────────────────────────────────────────
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only

from transformers import TrainingArguments
from trl import SFTTrainer

# ═════════════════════════════════════════════════════════════════════════
# 配置
# ═════════════════════════════════════════════════════════════════════════

# --- 模型 ---
MODEL_NAME = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 1024                # 4GB VRAM 减半，数据 avg=649 够用
LOAD_IN_4BIT = True
DTYPE = None                         # 自动选择 fp16/bf16

# --- LoRA ---
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0                     # 设为 0 有优化加速
LORA_BIAS = "none"
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# --- 训练 ---
BATCH_SIZE = 1                       # 4GB VRAM 最小值
GRADIENT_ACCUMULATION_STEPS = 8      # 有效 batch = 1×8 = 8，保持不变
MAX_STEPS = -1                       # -1 表示用 num_train_epochs
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
WARMUP_STEPS = 50
LR_SCHEDULER = "cosine"
OPTIMIZER = "adamw_8bit"             # 8-bit 优化器省显存
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 0.3                  # 梯度裁剪，防止 outlier 导致 loss 爆炸
LOGGING_STEPS = 10
SAVE_STEPS = 200
SAVE_TOTAL_LIMIT = 3                 # 只保留最近 3 个 checkpoint
EVAL_STEPS = 100                     # 每 100 步验证一次

# --- 数据 ---
DATASET_NAME = "FreedomIntelligence/medical-o1-reasoning-SFT"
DATASET_CONFIG = "en"                # 英文子集
DATASET_SPLIT = "train"
EVAL_SPLIT_RATIO = 0.05              # 从训练集切 5% 做验证

# --- 输出 ---
OUTPUT_DIR = "./output_qwen05b_medical_o1"
HUB_MODEL_ID = None                  # 如要上传 HF 请填写 "your-namespace/model-name"
SAVE_MERGED_16BIT = True             # 训练完 merge 成 16-bit 权重


# ═════════════════════════════════════════════════════════════════════════
# 数据预处理
# ═════════════════════════════════════════════════════════════════════════

def format_medical_entry(example):
    """
    将 medical-o1-reasoning-SFT 的每条数据格式化为对话模板。

    数据结构:
      - Question:     str, 医学问题
      - Complex_CoT:  str, 详细推理链
      - Response:     str, 最终答案

    输出格式 (ChatML / Qwen2.5-Instruct):
      <|im_start|>user
      {Question}<|im_end|>
      <|im_start|>assistant
      <think>{Complex_CoT}</think>{Response}<|im_end|>

    注意: 我们保留 Complex_CoT 作为 <think> 推理过程，
          让模型学会先推理再回答。
    """
    question = example["Question"]
    cot = example["Complex_CoT"]
    answer = example["Response"]

    # 构建对话
    conversation = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": f"<think>{cot}</think>\n{answer}"},
    ]
    return {"conversation": conversation}


def preprocess_dataset(dataset, tokenizer):
    """对整个数据集执行格式化和 tokenize。"""
    # 1) 格式化为对话
    dataset = dataset.map(format_medical_entry, remove_columns=dataset.column_names)

    # 2) 应用 Qwen2.5 的 chat_template
    dataset = dataset.map(
        lambda x: {
            "text": tokenizer.apply_chat_template(
                x["conversation"],
                tokenize=False,
                add_generation_prompt=False,
            )
        },
        remove_columns=["conversation"],
    )
    return dataset


# ═════════════════════════════════════════════════════════════════════════
# 主流程
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="从指定 checkpoint 恢复训练")
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS,
                        help="训练步数，覆盖 NUM_EPOCHS")
    parser.add_argument("--num_epochs", type=float, default=NUM_EPOCHS,
                        help="训练轮数")
    args = parser.parse_args()

    # ── 打印配置 ──
    print("=" * 60)
    print("    Qwen2.5-0.5B + Unsloth + QLoRA  医疗问答微调")
    print("=" * 60)
    print(f"  模型:     {MODEL_NAME}")
    print(f"  数据:     {DATASET_NAME} ({DATASET_CONFIG})")
    print(f"  批次:     {BATCH_SIZE} × {GRADIENT_ACCUMULATION_STEPS} (有效={BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    print(f"  精度:     4-bit QLoRA")
    print(f"  显存:     RTX 3050 (4GB)")
    print(f"  输出:     {OUTPUT_DIR}")
    print("=" * 60)

    # ── 1. 加载模型与 tokenizer ──
    print("\n[1/5] 加载模型...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
        device_map="auto",           # 自动分配到 GPU
    )

    # ── 2. 配置 LoRA ──
    print("\n[2/5] 配置 LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias=LORA_BIAS,
        use_gradient_checkpointing="unsloth",   # Unsloth 优化版
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # 获取 Qwen2.5 的 chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen-2.5",
    )

    # ── 3. 加载数据集 ──
    print(f"\n[3/5] 加载数据集 {DATASET_NAME} ({DATASET_CONFIG})...")
    full_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    print(f"  原始样本数: {len(full_dataset)}")

    # 数据预览
    print("\n  数据样例 (第一条):")
    print(f"    Question:  {full_dataset[0]['Question'][:120]}...")
    print(f"    Complex_CoT: {full_dataset[0]['Complex_CoT'][:120]}...")
    print(f"    Response:  {full_dataset[0]['Response'][:120]}...")

    # 划分训练/验证集
    split_dataset = full_dataset.train_test_split(
        test_size=EVAL_SPLIT_RATIO, seed=3407
    )
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    print(f"  训练集: {len(train_dataset)} | 验证集: {len(eval_dataset)}")

    # 预处理
    train_dataset = preprocess_dataset(train_dataset, tokenizer)
    eval_dataset = preprocess_dataset(eval_dataset, tokenizer)
    print(f"  预处理后: 训练 {len(train_dataset)} | 验证 {len(eval_dataset)}")

    # 预估 token 分布
    for name, ds in [("训练", train_dataset), ("验证", eval_dataset)]:
        lengths = [len(tokenizer.encode(t)) for t in ds["text"]]
        print(f"  {name} Token 长度:  avg={sum(lengths)/len(lengths):.0f}  "
              f"max={max(lengths)}  min={min(lengths)}")

    # ── 4. 配置训练器 ──
    print("\n[4/5] 配置 SFTTrainer...")

    # 判断是否支持 bf16
    use_bf16 = is_bfloat16_supported()
    print(f"  BF16 支持: {use_bf16}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type=LR_SCHEDULER,
        optim=OPTIMIZER,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        save_strategy="steps",              # 每 200 步保存，防止 OOM 断电白训
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        fp16=not use_bf16,
        bf16=use_bf16,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        dataloader_num_workers=0,                # Windows spawn 模式设 0 避免 CUDA 死锁
        ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
        seed=3407,
        data_seed=3407,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
    )

    # 关键: 只对 assistant 部分的响应计算 loss
    # 让模型学会生成 <think>推理</think>答案 这部分
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user",
        response_part="<|im_start|>assistant",
    )

    # ── 5. 开始训练 ──
    print("\n[5/5] 开始训练...")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 记录起始显存
    torch.cuda.reset_peak_memory_stats()
    start_gpu_memory = torch.cuda.max_memory_reserved() / 1024**3

    torch.cuda.empty_cache()          # 清理碎片显存
    trainer_stats = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # ── 训练完成统计 ──
    print("=" * 60)
    print("  训练完成!")
    print(f"  结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  训练耗时: {trainer_stats.metrics['train_runtime']:.2f}s "
          f"({trainer_stats.metrics['train_runtime']/60:.2f}min)")
    print(f"  最终 loss: {trainer_stats.metrics.get('train_loss', 'N/A'):.4f}")
    used_memory = torch.cuda.max_memory_reserved() / 1024**3
    print(f"  峰值显存: {used_memory:.3f} GB")
    print(f"  训练峰值显存: {used_memory - start_gpu_memory:.3f} GB")
    print("=" * 60)

    # ── 保存模型 ──
    print("\n保存模型...")

    # 保存 LoRA 适配器 (体积小，方便继续训练)
    model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapter"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapter"))
    print(f"  LoRA 适配器已保存至: {OUTPUT_DIR}/lora_adapter")

    # 可选: merge 成 16-bit 完整权重 (可用于推理/上传)
    if SAVE_MERGED_16BIT:
        print("  合并 LoRA → 完整 16-bit 权重...")
        merged_dir = os.path.join(OUTPUT_DIR, "merged_16bit")
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        print(f"  合并权重已保存至: {merged_dir}")

    # 可选: 导出 GGUF (用于 Ollama/llama.cpp)
    # model.save_pretrained_merged(OUTPUT_DIR + "/gguf", tokenizer, save_method="gguf")

    # 可选: 上传到 Hugging Face
    if HUB_MODEL_ID:
        print(f"  上传至 Hugging Face: {HUB_MODEL_ID}...")
        model.push_to_hub_merged(HUB_MODEL_ID, tokenizer, save_method="merged_16bit")
        print("  上传完成!")

    print("\n✅ 全部完成!")


if __name__ == "__main__":
    main()
