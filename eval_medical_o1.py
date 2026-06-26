"""
=============================================================================
 评估脚本: Qwen2.5-0.5B + QLoRA 医疗问答
=============================================================================
 用法:
   # 评估最新 LoRA 适配器
   python eval_medical_o1.py

   # 指定 checkpoint
   python eval_medical_o1.py --checkpoint ./output_qwen05b_medical_o1/checkpoint-200

   # 采样 10 条 + 输出到文件
   python eval_medical_o1.py --num_samples 10 --output eval_results.txt
=============================================================================
"""

import os
import re
import torch
import argparse
from datasets import load_dataset
from transformers import TextStreamer

from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template

# ── 配置（与训练脚本保持一致）──
MODEL_NAME = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048
DATASET_NAME = "FreedomIntelligence/medical-o1-reasoning-SFT"
DATASET_CONFIG = "en"
OUTPUT_DIR = "./output_qwen05b_medical_o1"


def extract_think_answer(text: str):
    """从模型输出中分离 <think> 推理和最终答案。"""
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        reasoning = think_match.group(1).strip()
        answer = text[think_match.end():].strip()
        return reasoning, answer
    return None, text.strip()


@torch.inference_mode()
def generate_answer(model, tokenizer, question: str,
                    max_new_tokens=512, temperature=0.1, top_p=0.9,
                    do_stream=False):
    """对单个问题生成回答。"""
    messages = [{"role": "user", "content": question}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to("cuda")

    if do_stream:
        text_streamer = TextStreamer(tokenizer, skip_prompt=True)
        _ = model.generate(
            input_ids=inputs, streamer=text_streamer,
            max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p,
            do_sample=True, pad_token_id=tokenizer.eos_token_id,
        )
        return None, None, "(已流式输出)"

    generated = model.generate(
        input_ids=inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=top_p,
        do_sample=True, pad_token_id=tokenizer.eos_token_id,
    )
    output = tokenizer.decode(generated[0][inputs.shape[1]:], skip_special_tokens=True)
    reasoning, answer = extract_think_answer(output)
    return reasoning, answer, output


def compute_keyword_overlap(pred: str, reference: str) -> float:
    """简单关键词重叠率，用于快速参考。"""
    pred_set = set(pred.lower().split()[:15])
    ref_set = set(reference.lower().split()[:15])
    if not ref_set:
        return 0.0
    return len(pred_set & ref_set) / len(ref_set)


def main():
    parser = argparse.ArgumentParser(description="评估微调后的医疗问答模型")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="LoRA 适配器路径 (默认使用 output dir 下的 lora_adapter)")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="评估样本数 (默认 5)")
    parser.add_argument("--output", type=str, default=None,
                        help="结果输出到文件 (可选)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # ── 确定适配器路径 ──
    adapter_path = args.checkpoint or os.path.join(OUTPUT_DIR, "lora_adapter")
    if not os.path.exists(adapter_path):
        print(f"❌ 找不到适配器: {adapter_path}")
        print(f"   请先运行 train_medical_o1.py 完成训练。")
        print(f"   或通过 --checkpoint 指定路径。")
        return

    print("=" * 60)
    print("  医疗问答模型评估")
    print("=" * 60)
    print(f"  基座模型:     {MODEL_NAME}")
    print(f"  LoRA 适配器:  {adapter_path}")
    print(f"  评估数据:     {DATASET_NAME} ({DATASET_CONFIG})")
    print(f"  样本数:       {args.num_samples}")
    print("=" * 60)

    # ── 1. 加载模型 + LoRA 适配器 ──
    print("\n[1/3] 加载模型 + LoRA 适配器...")
    # FastLanguageModel.from_pretrained 能自动识别 adapter 路径,
    # 读取 adapter_config.json 中引用的基座模型后加载 LoRA 权重,
    # 同时保留 Unsloth 的优化内核（比 PeftModel.from_pretrained 更快）
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    # chat template
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    # ── 2. 加载数据集，使用与训练一致的 5% 验证集划分 ──
    print("\n[2/3] 加载评估数据...")
    # 与 train_medical_o1.py 中 train_test_split(test_size=0.05, seed=3407) 一致
    full_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
    _, eval_dataset = full_dataset.train_test_split(test_size=0.05, seed=3407)
    test_samples = eval_dataset.select(
        range(min(args.num_samples, len(eval_dataset)))
    )
    print(f"  取了 {len(test_samples)} 条测试样本 (从 5% 验证集采样，训练未见过的)")

    # ── 3. 逐条评估 ──
    print(f"\n[3/3] 开始评估...\n")

    results = []
    for i, sample in enumerate(test_samples):
        question = sample["Question"]
        gt_cot = sample["Complex_CoT"]
        gt_answer = sample["Response"]

        print(f"{'─' * 60}")
        print(f"  [{i+1}/{args.num_samples}]")
        print(f"{'─' * 60}")
        print(f"  📝 问题:\n    {question[:200]}{'...' if len(question) > 200 else ''}")
        print()

        # ── 模型生成 ──
        print(f"  🤖 模型生成:\n")
        reasoning, answer, raw_output = generate_answer(
            model, tokenizer, question,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        if reasoning:
            print(f"     🧠 推理链:\n       {reasoning[:350]}{'...' if len(reasoning) > 350 else ''}")
            print(f"\n     💡 答案:\n       {answer[:200]}{'...' if len(answer) > 200 else ''}")
        else:
            print(f"     {answer[:400]}{'...' if len(answer) > 400 else ''}")

        # ── 标准答案 ──
        print(f"\n  ✅ 标准:")
        print(f"     🧠 推理链:\n       {gt_cot[:350]}{'...' if len(gt_cot) > 350 else ''}")
        print(f"\n     💡 答案:\n       {gt_answer[:200]}{'...' if len(gt_answer) > 200 else ''}")
        print()

        # ── 关键词重叠 ──
        pred_text = answer or raw_output
        overlap_keywords = compute_keyword_overlap(pred_text, gt_answer)
        overlap_cot = compute_keyword_overlap(pred_text, gt_cot) if not reasoning else \
                      compute_keyword_overlap(reasoning, gt_cot)
        print(f"  📊 关键词重叠 (答案): {overlap_keywords:.0%}")
        print(f"  📊 关键词重叠 (推理): {overlap_cot:.0%}")

        results.append({
            "question": question,
            "predicted_reasoning": reasoning,
            "predicted_answer": answer,
            "predicted_raw": raw_output,
            "ground_truth_cot": gt_cot,
            "ground_truth_answer": gt_answer,
            "keyword_overlap_answer": overlap_keywords,
            "keyword_overlap_cot": overlap_cot,
        })

    # ── 汇总 ──
    avg_answer_overlap = sum(r["keyword_overlap_answer"] for r in results) / len(results)
    avg_cot_overlap = sum(r["keyword_overlap_cot"] for r in results) / len(results)

    print(f"\n{'=' * 60}")
    print(f"  评估汇总")
    print(f"{'=' * 60}")
    print(f"  样本数:                {len(results)}")
    print(f"  平均关键词重叠 (答案):  {avg_answer_overlap:.0%}")
    print(f"  平均关键词重叠 (推理):  {avg_cot_overlap:.0%}")
    print(f"  ⚠️ 关键词重叠仅作快速参考，请逐条人工判断回答质量")
    print(f"  💡 如果模型回答和标准答案意思一致但措辞不同，重叠率会偏低")
    print(f"{'=' * 60}")

    # 保存详细结果
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for i, r in enumerate(results):
                f.write(f"{'=' * 60}\n")
                f.write(f" 样本 #{i+1}\n")
                f.write(f"{'=' * 60}\n")
                f.write(f"【问题】\n{r['question']}\n\n")
                f.write(f"【模型推理】\n{r['predicted_reasoning'] or 'N/A'}\n\n")
                f.write(f"【模型答案】\n{r['predicted_answer']}\n\n")
                f.write(f"【标准推理】\n{r['ground_truth_cot']}\n\n")
                f.write(f"【标准答案】\n{r['ground_truth_answer']}\n\n")
                f.write(f"关键词重叠: 答案={r['keyword_overlap_answer']:.0%}, "
                       f"推理={r['keyword_overlap_cot']:.0%}\n\n")
        print(f"  详细结果已保存到: {args.output}")

    print("\n✅ 评估完成!")


if __name__ == "__main__":
    main()
