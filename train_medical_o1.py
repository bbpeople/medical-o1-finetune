"""
=============================================================================
 Qwen2.5-Instruct + Unsloth + QLoRA 医疗问答微调
=============================================================================
 基座: unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit (默认；可改 BASE_SIZE / MODEL_NAME)
 数据: FreedomIntelligence/medical-o1-reasoning-SFT (英文子集)
 硬件: RTX 3050 (4GB VRAM) — 1.5B 需 MAX_SEQ_LENGTH=512 才能跑通
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

from transformers import TrainingArguments, TrainerCallback
from trl import SFTTrainer


# ═════════════════════════════════════════════════════════════════════════
# 自定义 Callback：绕过 Unsloth pickle 崩溃，手动保存 checkpoint
# ═════════════════════════════════════════════════════════════════════════
class ManualSaveCallback(TrainerCallback):
    """每 save_steps 步用 model.save_pretrained() 保存 checkpoint，不经过 pickle。

    保存内容（全部绕过 HF Trainer 原生 _save_checkpoint，规避 Unsloth pickle 崩溃）：
      - adapter_model.safetensors + adapter_config.json  (LoRA 权重，save_pretrained)
      - trainer_state.json        (HF TrainerState，标准 JSON，无 pickle)
      - optimizer.pt / scheduler.pt  (尽量写，失败则降级为近似续训 — 见 on_step_end)
      - rng_state.pth             (尽量写，失败则跳过)

    resume 时把 checkpoint 路径传给 trainer.train(resume_from_checkpoint=...)，
    HF 的 _load_optimizer_and_scheduler 只在 optimizer.pt + scheduler.pt 同时存在时
    才恢复优化器/调度器状态，否则优雅降级（只恢复 adapter 权重 + global_step，
    optimizer 从零初始化 = 近似续训）。这样可以在"optimizer 是否能被 torch.save"
    不确定时，先尝试完整保存，崩了自动退化为安全路径。
    """

    def __init__(self, save_steps, output_dir, save_total_limit=3):
        self.save_steps = save_steps
        self.output_dir = output_dir
        self.save_total_limit = save_total_limit
        self._checkpoint_dirs = []
        self._last_save_step = 0

    @staticmethod
    def _gather_rng_states():
        """收集 CPU/Python/CUDA 三种 RNG 状态，供 resume 复现随机性。"""
        import random
        rng = {
            "python": random.getstate(),
            "cpu": torch.get_rng_state(),
        }
        try:
            if torch.cuda.is_available():
                rng["cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
        return rng

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0:
            return control
        # 检查是否到了保存步数
        if state.global_step - self._last_save_step >= self.save_steps:
            self._last_save_step = state.global_step
            ckpt_dir = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            print(f"\n  💾 手动保存 checkpoint → {ckpt_dir}")
            # 从 kwargs 或 trainer 属性获取 model / tokenizer / trainer
            trainer = kwargs.get("trainer")
            model = kwargs.get("model") or (trainer.model if trainer else None)
            tokenizer = (kwargs.get("tokenizer")
                         or getattr(trainer, "tokenizer", None)) if trainer else None

            # 1) LoRA adapter 权重（save_pretrained，不走 pickle）
            if model is not None:
                model.save_pretrained(ckpt_dir)
            if tokenizer is not None:
                tokenizer.save_pretrained(ckpt_dir)

            # 2) trainer_state.json —— 用 HF 自带 TrainerState 序列化，零拼装风险。
            #    必须有它，resume 时 HF 才能读出 global_step / epoch 正确续训。
            try:
                state.save_to_json(os.path.join(ckpt_dir, "trainer_state.json"))
                print(f"  ✅ trainer_state.json (global_step={state.global_step})")
            except Exception as e:
                print(f"  ⚠️ trainer_state.json 保存失败: {e}（resume 将无法恢复步数，从 0 开始）")

            # 3) optimizer.pt + scheduler.pt —— 尽量写，失败则降级。
            #    8-bit 优化器 (bitsandbytes / paged_adamw_8bit) 的 state_dict 有时无法被
            #    torch.save/pickle；崩了就跳过，HF resume 会自动走近似续训路径。
            opt_ok = self._try_save_optimizer_scheduler(trainer, ckpt_dir)
            if not opt_ok:
                print("  ⚠️ optimizer.pt/scheduler.pt 未保存 → resume 将退化为【近似续训】")
                print("     （仅恢复 LoRA 权重 + 学习率位置，优化器动量重置）")

            # 4) rng_state.pth —— 尽量写，复现随机性（失败仅跳过，不影响续训）。
            try:
                torch.save(self._gather_rng_states(),
                           os.path.join(ckpt_dir, "rng_state.pth"))
            except Exception as e:
                print(f"  ⚠️ rng_state.pth 保存失败: {e}（跳过，不影响续训）")

            self._checkpoint_dirs.append(ckpt_dir)
            print(f"  ✅ checkpoint-{state.global_step} 保存成功")
            # 清理旧 checkpoint
            while len(self._checkpoint_dirs) > self.save_total_limit:
                old_dir = self._checkpoint_dirs.pop(0)
                import shutil
                shutil.rmtree(old_dir, ignore_errors=True)
                print(f"  🗑️ 清理旧 checkpoint: {old_dir}")
            print()
        return control

    @staticmethod
    def _try_save_optimizer_scheduler(trainer, ckpt_dir):
        """尝试保存 optimizer.pt + scheduler.pt。成功返回 True，任一失败返回 False。"""
        if trainer is None or getattr(trainer, "optimizer", None) is None:
            return False
        try:
            torch.save(trainer.optimizer.state_dict(),
                       os.path.join(ckpt_dir, "optimizer.pt"))
            torch.save(trainer.lr_scheduler.state_dict()
                       if trainer.lr_scheduler is not None else {},
                       os.path.join(ckpt_dir, "scheduler.pt"))
            print("  ✅ optimizer.pt + scheduler.pt")
            return True
        except Exception as e:
            print(f"  ⚠️ optimizer/scheduler 保存失败: {e}")
            # 清理可能写了一半的文件，避免半成品误导 resume
            for f in ("optimizer.pt", "scheduler.pt"):
                p = os.path.join(ckpt_dir, f)
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            return False


# ═════════════════════════════════════════════════════════════════════════
# 配置
# ═════════════════════════════════════════════════════════════════════════

# --- 模型 ---
BASE_SIZE = "1.5B"                   # 当前基座规格，用于输出目录命名与打印
MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 512                 # RTX 3050 4GB 跑 1.5B 必须 512；训练时长样本会被截断
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
# 输出目录按基座规格命名，避免不同基座的 checkpoint / LoRA 互相覆盖
OUTPUT_DIR = f"./output_qwen{BASE_SIZE.lower().replace('.', '')}_medical_o1"  # e.g. output_qwen15b_medical_o1
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
    """对整个数据集执行格式化和 tokenize。

    关键: 生成 chat template 文本后，对每条样本做【硬截断到 MAX_SEQ_LENGTH】。
    原因: Unsloth SFTTrainer 在 transformers 5.5 下，当样本 token 数 > max_seq_length
    时并不会在 collator 里自动截断，而是原样送进 batch；fused CE loss 的 chunked
    计算会把 logits 切成 chunk（与 labels 长度不一致），报
    `ValueError: Expected input batch_size (N) to match target batch_size (M)`。
    硬截断在数据侧喂入 ≤ MAX_SEQ_LENGTH 的文本，从根上避免该不匹配。
    （副作用: 超 MAX_SEQ_LENGTH 的样本其 assistant 答案/推理尾部会被砍掉，
     labels 尾部变 -100；seq=512 下这是可接受代价。）
    """
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

    # 3) 硬截断到 MAX_SEQ_LENGTH：tokenize → 取前 MAX_SEQ_LENGTH → decode
    #    这样无论 SFTTrainer collator 是否再截断，喂入的文本都不会超长。
    _seq_limit = MAX_SEQ_LENGTH

    def _truncate_text(example):
        ids = tokenizer.encode(example["text"])
        if len(ids) > _seq_limit:
            ids = ids[:_seq_limit]
            example["text"] = tokenizer.decode(ids)
        return example

    dataset = dataset.map(_truncate_text)
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
    print(f"    Qwen2.5-{BASE_SIZE} + Unsloth + QLoRA  医疗问答微调")
    print("=" * 60)
    print(f"  模型:     {MODEL_NAME}")
    print(f"  数据:     {DATASET_NAME} ({DATASET_CONFIG})")
    print(f"  批次:     {BATCH_SIZE} × {GRADIENT_ACCUMULATION_STEPS} (有效={BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    print(f"  精度:     4-bit QLoRA")
    print(f"  序列:     {MAX_SEQ_LENGTH}")
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

    # 如果指定了 checkpoint，按以下策略恢复训练：
    #   ✅ 新 checkpoint（含 trainer_state.json）：保留路径传给 trainer.train()，
    #      由 HF 原生 _load_from_checkpoint + _load_optimizer_and_scheduler 处理：
    #        - adapter 权重自动 load（PEFT 标准路径）
    #        - 若 optimizer.pt + scheduler.pt 同时存在 → 完整无偏续训
    #        - 若不存在 → 优雅降级：只恢复 adapter + global_step，optimizer 重置（近似续训）
    #   ⚠️ 旧 checkpoint（只有 training_state.pt，无 trainer_state.json）：
    #      HF 无法读出 global_step，回退到「手动加载 LoRA 权重 + 从 0 训练」的旧行为。
    if args.resume_from_checkpoint is not None:
        ckpt_path = args.resume_from_checkpoint
        has_trainer_state = os.path.exists(os.path.join(ckpt_path, "trainer_state.json"))
        adapter_file = os.path.join(ckpt_path, "adapter_model.safetensors")

        if has_trainer_state:
            print(f"\n  🔁 检测到新式 checkpoint（含 trainer_state.json）：{ckpt_path}")
            opt_present = os.path.exists(os.path.join(ckpt_path, "optimizer.pt"))
            sched_present = os.path.exists(os.path.join(ckpt_path, "scheduler.pt"))
            if opt_present and sched_present:
                print("     ✅ optimizer.pt + scheduler.pt 存在 → 将执行完整无偏续训")
            else:
                print("     ⚠️ 缺 optimizer.pt/scheduler.pt → 将退化为【近似续训】")
                print("        （恢复 LoRA 权重 + 学习率位置，优化器动量重置）")
            print("     → 保留路径，交由 trainer.train(resume_from_checkpoint=...) 原生处理")
            # 不清空 args.resume_from_checkpoint，HF 在 train() 中读取它
        else:
            print(f"\n  ⚠️ 旧式 checkpoint（无 trainer_state.json）：{ckpt_path}")
            print("     HF resume 无法读出训练步数，回退到「加载 LoRA 权重 + 从 0 训练」。")
            if os.path.exists(adapter_file):
                print(f"     手动加载 LoRA 权重...")
                from safetensors.torch import load_file
                state_dict = load_file(adapter_file)
                model.load_state_dict(state_dict, strict=False)
                print(f"  ✅ LoRA 权重已加载（{len(state_dict)} 个张量）")
            else:
                print(f"  ⚠️ 未找到 adapter_model.safetensors，以随机初始化继续")
            # 清空，防止 trainer.train(resume_from_checkpoint=...) 去找一个无 trainer_state 的目录
            args.resume_from_checkpoint = None

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
        save_strategy="no",                # 用自定义 callback 保存（见下方），规避 pickle 崩溃
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

    # 注册自定义 checkpoint 保存 callback
    trainer.add_callback(ManualSaveCallback(
        save_steps=SAVE_STEPS,
        output_dir=OUTPUT_DIR,
        save_total_limit=SAVE_TOTAL_LIMIT,
    ))

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

    # Fail-safe：HF 原生 resume 在极少数情况下可能抛错（如 adapter_config.json 缺失、
    # optimizer state 版本不匹配等）。捕获后回退到「手动加载 LoRA + 从 0 训」并重试一次，
    # 保证训练能继续推进，而不是直接崩在起点（此时尚未真正开始训练，无进度损失风险）。
    try:
        trainer_stats = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    except Exception as resume_err:
        if args.resume_from_checkpoint is None:
            raise                # 真正的新训练都崩了，应如实抛出，不要吞掉
        print(f"\n  ⚠️ HF 原生 resume 失败: {resume_err}")
        print("     回退到「手动加载 LoRA 权重 + 从 0 训练」并重试 ...")
        ckpt_path = args.resume_from_checkpoint
        adapter_file = os.path.join(ckpt_path, "adapter_model.safetensors")
        loaded = False
        if os.path.exists(adapter_file):
            try:
                from safetensors.torch import load_file
                sd = load_file(adapter_file)
                model.load_state_dict(sd, strict=False)
                print(f"  ✅ LoRA 权重已手动加载（{len(sd)} 个张量）")
                loaded = True
            except Exception as e2:
                print(f"  ⚠️ 手动加载 LoRA 也失败: {e2}（将以随机初始化重训）")
        else:
            print(f"  ⚠️ checkpoint 无 adapter_model.safetensors，随机初始化重训")
        # 注：此处不重建 optimizer/scheduler。HF Trainer.train(resume_from_checkpoint=None)
        # 会从 global_step=0 重新计数（lr 走完整 warmup+cosine），最坏情况下 resume 失败前
        # 可能已半写入 optimizer 动量，这只会让早期几步的更新轨迹略有偏差，不构成 bug。
        torch.cuda.empty_cache()
        trainer_stats = trainer.train(resume_from_checkpoint=None)
        if loaded:
            print("  ⚠️ 本次为【从 0 重训 + LoRA 权重预热】，不等于原步数无偏续训")

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
