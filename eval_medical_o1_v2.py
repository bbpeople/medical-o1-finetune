"""
=============================================================================
 医疗问答评估 v2 — 选择题命中率 + 开放问答关键词重叠
=============================================================================
 基座/产物: output_qwen15b_medical_o1/merged_16bit (Qwen2.5-1.5B-Instruct, bf16)
 加载方式: AutoModelForCausalLM(完整 16-bit bf16)，不依赖 Unsloth/4-bit，防中断
 解码: 贪心(do_sample=False)，确定性、可复现
 数据: FreedomIntelligence/medical-o1-reasoning-SFT (en)，5% holdout(seed=3407)

 两个指标:
   A) 选择题选项命中率(严口径): 模型生成字母 == 参考 Response 抽出的字母
   B) 开放问答关键词重叠(辅助，与 0.5B 历史基线对比)

 防中断: 每条 try/except + OOM 降级、每 10 条 flush JSON、--resume 断点续跑。

 用法:
   D:\\anaconda\\envs\\kcsj_new\\python.exe eval_medical_o1_v2.py
   ... --resume   (断点续跑)
详设计见 docs/superpowers/specs/2026-07-08-medical-eval-v2-design.md
=============================================================================
"""

import os
import re
import json
import argparse
import time
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ── 配置(与训练脚本一致)──────────────────────────────────────────────
DATASET_NAME = "FreedomIntelligence/medical-o1-reasoning-SFT"
DATASET_CONFIG = "en"
EVAL_SPLIT_RATIO = 0.05
SPLIT_SEED = 3407            # 与 train_medical_o1.py 完全一致，保证同一个 holdout
MAX_SEQ_LENGTH = 512         # 与训练一致


# ═══════════════════════════════════════════════════════════════════════
# 答案字母抽取
# ═══════════════════════════════════════════════════════════════════════

# 题干选项识别: A) / A. / A: 这种独立选项标号
QUESTION_OPT_RE = re.compile(r'\b([A-E])\s*[\).:]\s', re.I)

# 答案字母抽取: 按优先级匹配
_ANSWER_PATTERNS = [
    re.compile(r'answer\s+is\s+\(?([A-E])\)?', re.I),
    re.compile(r'Answer\s*:\s*\(?([A-E])\)?', re.I),
    re.compile(r'answer\s+is\s+option\s+\(?([A-E])\)?', re.I),
    re.compile(r'option\s+\(?([A-E])\)?', re.I),
    re.compile(r'corresponds?\s+to\s+option\s+\(?([A-E])\)?', re.I),
    re.compile(r'the\s+correct\s+answer\s+is\s+\(?([A-E])\)?', re.I),
    re.compile(r'correct\s+option\s+is\s+\(?([A-E])\)?', re.I),
]
# 兜底: 文本里第一个独立字母 (最后再用,且必须 ∈ 题干选项集)
_FALLBACK_LETTER_RE = re.compile(r'\b([A-E])\b')


def question_options(question: str):
    """从题干抽出的选项字母集合,如 {'A','B','C','D'}。"""
    found = QUESTION_OPT_RE.findall(question)
    return set(o.upper() for o in found)


def extract_answer_letter(text: str, valid_letters=None):
    """从模型生成/参考 Response 文本抽答案字母。

    返回 (letter, source) 或 (None, None)。
    letter 必须在 valid_letters(题干选项)内,否则视为无效(返回 None)。
    source: 'pattern' | 'fallback' | None
    """
    for pat in _ANSWER_PATTERNS:
        m = pat.search(text)
        if m:
            letter = m.group(1).upper()
            if valid_letters is None or letter in valid_letters:
                return letter, "pattern"
            # pattern 命中但越界,直接返回 None(强信号被否决)
            return None, "pattern_out_of_range"
    # 兜底: 第一个独立字母
    if valid_letters:
        for m in _FALLBACK_LETTER_RE.finditer(text):
            letter = m.group(1).upper()
            if letter in valid_letters:
                return letter, "fallback"
    return None, None


# ═══════════════════════════════════════════════════════════════════════
# 关键词重叠(沿用原 eval 脚本语义,与 0.5B 基线可直接对比)
# ═══════════════════════════════════════════════════════════════════════

def compute_keyword_overlap(pred: str, reference: str) -> float:
    pred_set = set(pred.lower().split()[:15])
    ref_set = set(reference.lower().split()[:15])
    if not ref_set:
        return 0.0
    return len(pred_set & ref_set) / len(ref_set)


def extract_think_answer(text: str):
    """Separate reasoning inside the think tags from the final answer."""
    pattern = "<think>(.*?)</think>"
    m = re.search(pattern, text, re.DOTALL)
    if m:
        reasoning = m.group(1).strip()
        answer = text[m.end():].strip()
        return reasoning, answer
    return None, text.strip()


# ═══════════════════════════════════════════════════════════════════════
# 推理
# ═══════════════════════════════════════════════════════════════════════

def build_prompt(tokenizer, question: str):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )


@torch.inference_mode()
def generate_answer(model, tokenizer, question: str, device,
                    max_new_tokens=512):
    """贪心解码,返回生成的续写文本(不含 prompt)。OOM 降级 256 重试一次。"""
    inputs = build_prompt(tokenizer, question).to(device)

    def _gen(mnt):
        out = model.generate(
            input_ids=inputs,
            max_new_tokens=mnt,
            do_sample=False,            # 贪心,确定性
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)

    try:
        text = _gen(max_new_tokens)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if device == "cuda":
            torch.cuda.empty_cache()
        text = _gen(max(64, max_new_tokens // 2))   # 降级
    finally:
        if device == "cuda":
            torch.cuda.empty_cache()
    return text


# ═══════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="医疗问答评估 v2 (命中率+重叠)")
    parser.add_argument("--model_dir", type=str,
                        default="output_qwen15b_medical_o1/merged_16bit")
    parser.add_argument("--num_open", type=int, default=50,
                        help="开放问答样本数(默认50;0=全量 holdout)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda 或 cpu")
    parser.add_argument("--resume", action="store_true",
                        help="从已有 JSON 续跑(跳过已完成题)")
    parser.add_argument("--out_prefix", type=str,
                        default="eval_15b_v2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA 不可用,自动切到 CPU")
        device = "cpu"

    torch.manual_seed(args.seed)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = f"{args.out_prefix}_{ts}.json"
    out_txt = f"{args.out_prefix}_{ts}.txt"

    print("=" * 64)
    print("  医疗问答评估 v2 (命中率 + 关键词重叠)")
    print("=" * 64)
    print(f"  模型:      {args.model_dir}")
    print(f"  设备:      {device}")
    print(f"  开放问答数: {args.num_open}")
    print(f"  max_new_tokens: {args.max_new_tokens}")
    print(f"  解码:      贪心(do_sample=False, 确定性可复现)")
    print("=" * 64)

    # ── 1. 加载模型 ──
    print("\n[1/4] 加载模型...")
    t0 = time.time()
    dtype = torch.bfloat16
    if device == "cpu":
        # CPU 上 bf16 也可,但某些 CPU 对 bf16 支持差;保留 bf16 以与权重一致
        pass
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device if device == "cuda" else "cpu",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    print(f"  ✅ 模型加载完成 ({time.time()-t0:.1f}s), 参数量 "
          f"{sum(p.numel() for p in model.parameters())/1e9:.4f}B")
    if device == "cuda":
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  显存: 已占用 {reserved:.2f} GB / {total:.2f} GB "
              f"({reserved/total*100:.0f}%)")
        if reserved / total > 0.7:
            print("  ⚠️ 显存占用已>70%,建议先清其他显存再跑,否则可能 OOM 中断")

    # ── 2. 加载数据 + 切 holdout(与训练同 seed)──
    print("\n[2/4] 加载数据集 + 切 5% holdout ...")
    full = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
    split = full.train_test_split(test_size=EVAL_SPLIT_RATIO, seed=SPLIT_SEED)
    eval_ds = split["test"]
    print(f"  holdout 验证集: {len(eval_ds)} 条")

    # 预扫: 标注每题是否选择题 + 参考字母
    mc_indices = []        # 选择题且能抽到参考字母(idx in eval_ds)
    ref_missing = []       # 选择题但参考字母抽不到(单独报告,不计分母)
    for i in range(len(eval_ds)):
        q = eval_ds[i]["Question"]
        r = eval_ds[i]["Response"]
        opts = question_options(q)
        if len(opts) >= 2:
            letter, src = extract_answer_letter(r, valid_letters=opts)
            if letter:
                mc_indices.append(i)
            else:
                ref_missing.append(i)
    print(f"  选择题(参考字母可抽): {len(mc_indices)} 条 → 命中率分母")
    print(f"  选择题(参考字母抽不到,剔除不计分母): {len(ref_missing)} 条")

    # 开放问答采样(全 holdout 的随机 N 条;含选择题也无所谓,B 指标是关键词重叠)
    open_n = len(eval_ds) if args.num_open <= 0 else args.num_open
    g = torch.Generator().manual_seed(args.seed)
    open_perm = torch.randperm(len(eval_ds), generator=g)[:open_n].tolist()
    print(f"  开放问答评估样本: {len(open_perm)} 条")

    # ── resume 恢复 ──
    results_mc, results_open = [], []
    done_mc_ids, done_open_ids = set(), set()
    if args.resume and os.path.exists(out_json):
        try:
            with open(out_json, "r", encoding="utf-8") as f:
                prev = json.load(f)
            results_mc = prev.get("mc", [])
            results_open = prev.get("open", [])
            done_mc_ids = {r["idx"] for r in results_mc}
            done_open_ids = {r["idx"] for r in results_open}
            print(f"  🔁 resume: 跳过已完成 mc {len(done_mc_ids)} 条, "
                  f"open {len(done_open_ids)} 条")
        except Exception as e:
            print(f"  ⚠️ resume 读取失败({e}),从头开始")

    # ── 3. 选择题命中率评估 ──
    print(f"\n[3/4] 选择题命中率评估 ({len(mc_indices)} 条)...")
    txt_fp = open(out_txt, "a", encoding="utf-8")
    def _flush_json():
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"mc": results_mc, "open": results_open,
                       "config": vars(args)}, f, ensure_ascii=False, indent=2)

    hit = 0; wrong = 0; no_letter = 0; err = 0
    per_letter = {c: {"hit": 0, "tot": 0} for c in "ABCDE"}
    for n, i in enumerate(mc_indices):
        if i in done_mc_ids:
            continue
        q = eval_ds[i]["Question"]
        gt_cot = eval_ds[i]["Complex_CoT"]
        gt_resp = eval_ds[i]["Response"]
        opts = question_options(q)
        ref_letter, _ = extract_answer_letter(gt_resp, valid_letters=opts)
        try:
            raw = generate_answer(model, tokenizer, q, device,
                                  max_new_tokens=args.max_new_tokens)
            reasoning, answer = extract_think_answer(raw)
            pred_letter, src = extract_answer_letter(answer or raw,
                                                     valid_letters=opts)
        except Exception as e:
            raw = f"<eval_error: {type(e).__name__}>"
            reasoning, answer = None, None
            pred_letter, src = None, "error"

        bucket = None
        if pred_letter is None and src == "error":
            bucket = "eval_error"; err += 1
        elif pred_letter is None:
            bucket = "no_letter"; no_letter += 1
        elif pred_letter == ref_letter:
            bucket = "hit"; hit += 1
            per_letter[ref_letter]["hit"] += 1
        else:
            bucket = "wrong"; wrong += 1
        per_letter[ref_letter]["tot"] += 1

        results_mc.append({
            "idx": i, "bucket": bucket, "ref_letter": ref_letter,
            "pred_letter": pred_letter, "extract_source": src,
            "question": q[:400], "ref_response": gt_resp[:400],
            "model_reasoning": (reasoning or "")[:400],
            "model_answer": (answer or raw)[:600],
        })
        status = {"hit":"✅命中","wrong":"❌答错","no_letter":"￤未给字母",
                  "eval_error":"￤评估失败"}[bucket]
        print(f"  [{n+1}/{len(mc_indices)}] idx={i} {status} "
              f"ref={ref_letter} pred={pred_letter} (src={src})")
        # flush 每 10 条
        if (n + 1) % 10 == 0:
            _flush_json()

    _flush_json()
    mc_total = hit + wrong + no_letter + err
    mc_acc = hit / mc_total * 100 if mc_total else 0.0
    print(f"\n  选择题: 命中 {hit}/{mc_total} = {mc_acc:.1f}% (严口径)")
    print(f"    答错 {wrong} | 未给字母 {no_letter} | 评估失败 {err} | "
          f"参考字母抽不到(剔除) {len(ref_missing)}")

    # ── 4. 开放问答关键词重叠 ──
    print(f"\n[4/4] 开放问答关键词重叠 ({len(open_perm)} 条)...")
    open_ans_overlap = []
    open_cot_overlap = []
    for n, i in enumerate(open_perm):
        if i in done_open_ids:
            continue
        q = eval_ds[i]["Question"]
        gt_cot = eval_ds[i]["Complex_CoT"]
        gt_resp = eval_ds[i]["Response"]
        try:
            raw = generate_answer(model, tokenizer, q, device,
                                  max_new_tokens=args.max_new_tokens)
            reasoning, answer = extract_think_answer(raw)
        except Exception as e:
            raw = f"<eval_error: {type(e).__name__}>"
            reasoning, answer = None, None
        pred_text = answer or raw
        ov_ans = compute_keyword_overlap(pred_text, gt_resp)
        ov_cot = compute_keyword_overlap(reasoning or pred_text, gt_cot)
        open_ans_overlap.append(ov_ans)
        open_cot_overlap.append(ov_cot)
        results_open.append({
            "idx": i, "overlap_answer": ov_ans, "overlap_cot": ov_cot,
            "question": q[:300], "ref_response": gt_resp[:300],
            "model_answer": (answer or raw)[:400],
            "model_reasoning": (reasoning or "")[:300],
        })
        print(f"  [{n+1}/{len(open_perm)}] idx={i} 答案重叠 {ov_ans:.0%} "
              f"推理重叠 {ov_cot:.0%}")
        if (n + 1) % 10 == 0:
            _flush_json()

    # 写 txt 人读版
    txt_fp.write("=" * 64 + "\n选择题命中明细 (严口径)\n" + "=" * 64 + "\n")
    for r in results_mc:
        txt_fp.write(f"\nidx={r['idx']} {r['bucket']} "
                     f"ref={r['ref_letter']} pred={r['pred_letter']}\n")
        txt_fp.write(f"【问题】{r['question']}\n")
        txt_fp.write(f"【参考】{r['ref_response']}\n")
        txt_fp.write(f"【模型】{r['model_answer']}\n")
        txt_fp.write("-" * 64 + "\n")
    txt_fp.write("\n" + "=" * 64 + "\n开放问答明细\n" + "=" * 64 + "\n")
    for r in results_open:
        txt_fp.write(f"\nidx={r['idx']} 答案重叠 {r['overlap_answer']:.0%} "
                     f"推理重叠 {r['overlap_cot']:.0%}\n")
        txt_fp.write(f"【问题】{r['question']}\n")
        txt_fp.write(f"【参考】{r['ref_response']}\n")
        txt_fp.write(f"【模型】{r['model_answer']}\n")
        txt_fp.write("-" * 64 + "\n")
    txt_fp.close()
    _flush_json()

    # ── 汇总 ──
    avg_ans = (sum(open_ans_overlap) / len(open_ans_overlap)) if open_ans_overlap else 0
    avg_cot = (sum(open_cot_overlap) / len(open_cot_overlap)) if open_cot_overlap else 0
    print("\n" + "=" * 64)
    print("  评估汇总")
    print("=" * 64)
    print(f"  选择题: 命中 {hit}/{mc_total} = {mc_acc:.1f}% (严口径)")
    print(f"    答错 {wrong} | 未给字母 {no_letter} | 评估失败 {err} | "
          f"参考抽不到(剔除) {len(ref_missing)}")
    letter_dist = "  ".join(
        f"{c} {per_letter[c]['hit']}/{per_letter[c]['tot']}"
        f"={per_letter[c]['hit']/per_letter[c]['tot']*100:.0f}%"
        if per_letter[c]['tot'] else f"{c} 0/0"
        for c in "ABCDE"
    )
    print(f"  每选项命中分布: {letter_dist}")
    print(f"  开放问答: 答案关键词重叠均值 {avg_ans*100:.1f}% "
          f"推理重叠 {avg_cot*100:.1f}% ({len(open_ans_overlap)} 条)")
    print(f"\n  结果文件: {out_json} / {out_txt}")
    print("✅ 评估完成!")


if __name__ == "__main__":
    main()
