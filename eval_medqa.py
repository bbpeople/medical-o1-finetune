"""
=============================================================================
 医疗选择题独立基准评估 —— MedQA (USMLE) 真金标准确率
=============================================================================
 目的:
   旧 eval_medical_o1_v2.py 的 30% 选择题命中率是"从 medical-o1 的 Response
   反抽字母当参考",没真金标、有噪声、置信区间 ±14%, 和随机猜(25%)无法区分。
   本脚本换成【带真金标的独立测试集】MedQA(USMLE), answer_idx 直接给对/错,
   拿到一个真正可信的专业选择题准确率。

 数据: GBaker/MedQA-USMLE-4-options (HF, 走 hf-mirror)
   split=test 1273 条; 字段: question / options(dict A-D) / answer / answer_idx(金标)
   题干是 USMLE 临床题, 英文, 四选项(A/B/C/D), 与 medical-o1 训练分布接近。

 口径与旧脚本(eval_medical_o1_v2.py)保持一致, 便于对照:
   - 加载: same merged_16bit 产物; GPU 时 4-bit NF4 量化(bf16 计算), 适配 4GB
   - 解码: 贪心(do_sample=False), 确定性可复现
   - 答案字母抽取: 复用同一组 _ANSWER_PATTERNS 正则 + fallback, 区分 'pattern'/'fallback'
   - 区别: 参考字母直接取数据集 answer_idx(真金标), 不再从模型/Response 反抽

 防中断: 每条 try/except + OOM 降级、每 5 条 flush JSON、--resume 断点续跑。
 用法:
   D:\\anaconda\\envs\\kcsj_new\\python.exe eval_medqa.py                # 抽样 200 条
   D:\\anaconda\\envs\\kcsj_new\\python.exe eval_medqa.py --num_mc 0     # 全量 1273 条
   ... --resume   (断点续跑)
=============================================================================
"""

import os
# 走国内镜像, 避免连 HF 主站超时(和 train/merge 脚本同策略)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 强制 stdout 行缓冲 —— cmd 窗口重定向/管道下默认全缓冲, 不加这行窗口会"看不到输出"
try:
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

print("eval_medqa starting...", flush=True)  # 第一行立即落地, 确认窗口/管道通畅

import re
import json
import argparse
import time
from datetime import datetime

import torch
import logging
# 抑制 "Both max_new_tokens and max_length seem to have been set" 等刷屏警告
# (Qwen2.5 generation_config 默认 max_length=32768 与 max_new_tokens 共存, 无害但每条生成刷一行)
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ── 配置 ────────────────────────────────────────────────────────────────
DATASET_NAME = "GBaker/MedQA-USMLE-4-options"
DATASET_SPLIT = "test"          # 1273 条, 带真金标 answer_idx


# ════════════════════════════════════════════════════════════════════════
# 答案字母抽取
# ════════════════════════════════════════════════════════════════════════
# 2026-07-10 修订: 保留原7个 pattern(其他人用过的口径, 向后兼容), 末尾【追加】
# 2 个宽松 pattern 兜底 —— 专抓模型实际常用但旧pattern漏掉的 "answer is:\n\nA)"
# 这类"冒号+换行+字母"格式。策略只增不减: 旧pattern先匹配成功不会走到追加的,
# 保证 idx=693 这类旧能抓的('option B')不丢; 只在旧全 miss 时才用宽松兜底抓新格式。
# 旧 pattern 改 strictly 会回归(曾让 idx=693 从 hit 变 no_letter), 故用追加而非替换。
_ANSWER_PATTERNS = [
    re.compile(r'answer\s+is\s+\(?([A-E])\)?', re.I),
    re.compile(r'Answer\s*:\s*\(?([A-E])\)?', re.I),
    re.compile(r'answer\s+is\s+option\s+\(?([A-E])\)?', re.I),
    re.compile(r'option\s+\(?([A-E])\)?', re.I),
    re.compile(r'corresponds?\s+to\s+option\s+\(?([A-E])\)?', re.I),
    re.compile(r'the\s+correct\s+answer\s+is\s+\(?([A-E])\)?', re.I),
    re.compile(r'correct\s+option\s+is\s+\(?([A-E])\)?', re.I),
    # ── 追加: 宽松兜底(仅当上面7个全miss) ──
    # 抓 "answer is: \n\nA)" / "correct answer is:\nA)" 这类冒号+换行+字母
    re.compile(r'(?:correct\s+)?answer\s+is\s*[:：]?\s*\(?([A-E])\)?', re.I),
    # 抓 "answer:\n\nA)" / "Answer:\nA)" (无is的冒号后换行再字母)
    re.compile(r'(?:the\s+)?answer\s*[:：]\s*\n*\(?([A-E])\)?', re.I),
    # ── 追加(2026-07-10): 模型实际最常用格式 "X) 选项文本" ──
    # probe6 复现 t900 那 5 条被判 fallback/wrong 的题: 模型闭合<\think>后,
    # answer 段是散文 "The most likely X is:\n\nB) Urinalysis..." / "**A) Defect**"
    # 直接以 "字母) 大写选项文本" 指认答案, 没有 "answer is" 语义前缀 -> 上面9个全miss,
    # fallback 用 \b[A-E]\b 抓散文里第一个裸字母(常是错位)。这里在裸字 fallback 之前
    # 加 "X) + 选项文本开头" 高约束 pattern: 要求 ) 紧跟空格+大写词(>=2字母), 排除
    # "A) I think" 这类口语; (?<!\w) 防止贴着 "seeA)" 词。仅作兜底, 语义pattern仍优先。
    re.compile(r'(?<!\w)\(?([A-E])\)?\)\s+(?:[A-Z][a-z]{2,}|[A-Z]{2,})'),
    # markdown bold: "**A) Defect in ATM gene**" (选项号被**包裹)
    re.compile(r'\*\*\(?\*?\s*\(?([A-E])\)?\s*\)\s*\*?\*?\s+[A-Z]'),
]
_FALLBACK_LETTER_RE = re.compile(r'\b([A-E])\b')


def extract_answer_letter(text: str, valid_letters=None):
    """与旧脚本同口径抽字母。valid_letters 限定选项集合(MedQA 恒为 {A,B,C,D})。

    返回 (letter, source): source ∈ {'pattern','fallback','pattern_out_of_range', None}
    """
    for pat in _ANSWER_PATTERNS:
        m = pat.search(text)
        if m:
            letter = m.group(1).upper()
            if valid_letters is None or letter in valid_letters:
                return letter, "pattern"
            return None, "pattern_out_of_range"
    if valid_letters:
        for m in _FALLBACK_LETTER_RE.finditer(text):
            letter = m.group(1).upper()
            if letter in valid_letters:
                return letter, "fallback"
    return None, None


def extract_think_answer(text: str):
    """Separate reasoning inside the think tags from the final answer.
    模型训练格式: <think>推理 答案。用 chr 构造标签字面量防渲染吞。
    """
    open_tag = '<think>'
    close_tag = '</think>'
    pat = re.compile(re.escape(open_tag) + r"(.*?)" + re.escape(close_tag), re.DOTALL)
    m = pat.search(text)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    return None, text.strip()


def build_mc_question(question: str, options: dict) -> str:
    """把 MedQA 的 question + options(dict) 拼成标准选择题题干:

        <question>

        A) <opt A>
        B) <opt B>
        C) <opt C>
        D) <opt D>

        Answer:

    让模型走它训练时学过的格式(慢思考 -> 选字母)。
    options 键按字母序输出(GBaker 版恒为 A/B/C/D)。
    """
    lines = [question.strip(), ""]
    for letter in sorted(options.keys()):
        lines.append(f"{letter}) {options[letter]}")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)


def build_prompt(tokenizer, question_text: str):
    """拼 chat 模板并 tokenize 为 (input_ids, attention_mask)。

    先用 apply_chat_template(tokenize=False) 拿到完整模板字符串, 再 tokenizer() 统一
    tokenize —— 这样【两条路径都稳】: transformers 路径原 apply_chat_template(tokenize=True)
    返 BatchEncoding 没问题, 但 unsloth 路径下 apply_chat_template(tokenize=True) 返回
    普通 Tensor(非 BatchEncoding), 访问 .input_ids 会 AttributeError。绕开 tokenize=True
    后两条路径都走 tokenizer() 得到标准 BatchEncoding。
    """
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question_text}],
        add_generation_prompt=True,
        tokenize=False,
    )
    enc = tokenizer(prompt_text, return_tensors="pt")
    return enc.input_ids, enc.attention_mask


@torch.inference_mode()
def generate_answer(model, tokenizer, question_text: str, device,
                    max_new_tokens=300):
    """贪心解码(OOM 自动降级一半 tokens 重试)。返回去掉 prompt 的生成文本。"""
    input_ids, attention_mask = build_prompt(tokenizer, question_text)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    prompt_len = input_ids.shape[1]

    def _gen(mnt):
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=mnt,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,   # kv cache, 避免 O(n^2) 重算历史 attention(4bit+LoRA 下尤其关键)
        )
        return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)

    try:
        text = _gen(max_new_tokens)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if device == "cuda":
            torch.cuda.empty_cache()   # 仅 OOM 降级时清, 回收碎片让重试有空间
        text = _gen(max(64, max_new_tokens // 2))
    # 注意: 正常路径【不】每条 empty_cache —— 连续推理下 empty_cache 会强制 GPU
    # 同步停顿并丢弃缓存, 拖慢逐条吞吐; 仅在上方 OOM 降级时才清。
    return text


@torch.inference_mode()
def generate_batch(model, tokenizer, question_texts, device, max_new_tokens,
                   pad_token_id):
    """批量贪心推理 —— 把 N 条 prompt 左 padding 对齐后一次性 generate,
    利用显存盈余喂饱 GPU(batch=1 时 GPU 利用率才 27%, batch=4 可拉到 >70%)。

    左 padding 是关键: pad 放在序列【前】面, 真实内容在末尾对齐, 每条 position_id
    自然正确(从对齐尾端起算), 生成的新 token 自然在各条末尾续接。
    与 generate_answer 同口径(贪心 + use_cache), OOM 时 batch 拆半重试。
    返回: List[str] —— 每条去掉各自 prompt 的生成文本(按 batch 内原顺序)。
    """
    if not question_texts:
        return []
    n = len(question_texts)

    def _run(batch_ids, batch_mask, mnt):
        # 左 padding 下必须显式给 position_ids: 比 attention_mask cumsum 是真实 token
        # 的从 0 起序号, pad 位置(attention_mask=0)的 cumsum-1 会被夹到 0(不参与attention)。
        # 不传的话部分模型(如 unsloth patched)不会自动跳 pad 算 position, 导致 batch 与
        # 单条结果不一致(probe3 实测题1 答案从 C 漂成 A)。
        position_ids = batch_mask.long().cumsum(-1) - 1
        position_ids = position_ids.masked_fill(batch_mask == 0, 0)
        out = model.generate(
            input_ids=batch_ids,
            attention_mask=batch_mask,
            position_ids=position_ids,
            max_new_tokens=mnt,
            do_sample=False,
            pad_token_id=pad_token_id,
            use_cache=True,
        )
        texts = []
        # 生成部分紧跟 aligned prompt 尾, 按列切片 [:, aligned_len:] 逐条 decode
        aligned_len = batch_mask.shape[1]
        for b in range(out.shape[0]):
            new_tokens = out[b][aligned_len:]
            texts.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
        return texts

    # 逐条 tokenize, 左 padding 对齐到最长 prompt
    encs = [build_prompt(tokenizer, qt) for qt in question_texts]
    aligned_len = max(ids.shape[1] for ids, _ in encs)
    padded = []
    for ids, am in encs:
        ids0, am0 = ids[0], am[0]            # 去 batch 维 → 1D
        pad_sz = aligned_len - ids0.shape[0]
        if pad_sz > 0:
            pad_ids = torch.full((pad_sz,), pad_token_id, dtype=ids0.dtype)
            pad_am = torch.zeros(pad_sz, dtype=am0.dtype)
            ids0 = torch.cat([pad_ids, ids0])
            am0 = torch.cat([pad_am, am0])
        padded.append((ids0.to(device), am0.to(device)))
    batch_ids = torch.stack([p[0] for p in padded])
    batch_mask = torch.stack([p[1] for p in padded])

    try:
        return _run(batch_ids, batch_mask, max_new_tokens)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if device == "cuda":
            torch.cuda.empty_cache()
        mid = n // 2
        if mid == 0:    # 单条仍 OOM, 降 max_new_tokens 救一把
            return [_run(batch_ids, batch_mask, max(64, max_new_tokens // 2))]
        return generate_batch(model, tokenizer, question_texts[:mid], device,
                              max_new_tokens, pad_token_id) + \
               generate_batch(model, tokenizer, question_texts[mid:], device,
                              max_new_tokens, pad_token_id)


# ════════════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="MedQA(USMLE) 真金标选择题准确率评估")
    parser.add_argument("--model_dir", type=str,
                        default="output_qwen15b_medical_o1/merged_16bit",
                        help="已 merge 的 16-bit 产物目录(默认评估方式)")
    parser.add_argument("--lora_adapter_dir", type=str, default=None,
                        help="若指定, 则直接加载【未量化基座+LoRA adapter】评测(跳过 merge)。"
                             "适配 adapter 用最新权重, 与 merged 产物同口径(均基座+LoRA 4bit推理)。"
                             "需同时指定 --base_model")
    parser.add_argument("--base_model", type=str, default=None,
                        help="--lora_adapter_dir 模式下用的未量化基座名/路径, 如 unsloth/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--num_mc", type=int, default=200,
                        help="抽样题数(默认200, 置信区间约+-7%%); 0=全量1273")
    parser.add_argument("--max_new_tokens", type=int, default=700,
                        help="生成上限(慢思考推理链长,需给足空间等其收尾选字母;15tok/s约46s/条)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda 或 cpu")
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                        help="GPU时用4bit量化加载,适配小显存")
    parser.add_argument("--no_load_in_4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--no_fast", action="store_true",
                        help="关闭 unsloth FastLanguageModel 推理加速, 退回纯 transformers 路径(慢, 兜底用)")
    parser.add_argument("--batch_size", type=int, default=0,
                        help="批量推理批大小(默认0=逐条, 口径可信)。>1 时用 generate_batch 左 padding "
                             "并行提速, 但实测 transformers generate 在 left-pad 下逐条生成位置续算错位, "
                             "会导致与单条结果不一致(probe3 实测题1 答案 C->A 漂移), 故默认关闭。仅速度优先、"
                             "可容忍小幅答案漂移时手动开启。")
    parser.add_argument("--resume", action="store_true",
                        help="从已有 JSON 续跑(跳过已完成题; 自动找 out_prefix_<ts>.json)")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="直接指定已有 JSON 路径续跑(避免 ts 变化导致 resume 找不到旧产物)。"
                             "续跑写入同一文件(idx 仍按 perm 顺序补齐尚未完成的条目)。")
    parser.add_argument("--out_prefix", type=str, default="eval_medqa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pollution_print", action="store_true", default=True,
                        help="打印前5条题干供人工核验是否见过原题")
    parser.add_argument("--no_pollution_print", dest="pollution_print", action="store_false")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA 不可用,自动切到 CPU")
        device = "cpu"

    torch.manual_seed(args.seed)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # resume_from 优先: 锁定续跑产物到指定文件(不再用新 ts), 避免旧 130 条丢失
    if args.resume_from:
        out_json = args.resume_from
        out_txt = out_json.replace(".json", ".txt")
    else:
        out_json = f"{args.out_prefix}_{ts}.json"
        out_txt = f"{args.out_prefix}_{ts}.txt"

    print("=" * 64)
    print("  MedQA(USMLE) 真金标选择题准确率评估")
    print("=" * 64)
    print(f"  模型:        {args.model_dir}")
    print(f"  数据:        {DATASET_NAME} (split={DATASET_SPLIT})")
    print(f"  设备:        {device}  (4bit={args.load_in_4bit and device=='cuda'})")
    print(f"  抽样题数:    {args.num_mc}  (0=全量1273)")
    print(f"  max_new_tok: {args.max_new_tokens}")
    print(f"  解码:        贪心(do_sample=False, 确定性可复现)")
    print("=" * 64)

    # ── 1. 加载模型 ──
    print("\n[1/4] 加载模型...")
    t0 = time.time()
    use_adapter = args.lora_adapter_dir is not None
    if use_adapter and not args.base_model:
        raise SystemExit("❌ 指定 --lora_adapter_dir 时必须同时给 --base_model (如 unsloth/Qwen2.5-0.5B-Instruct)")
    # ── unsloth 加速开关: adapter 模式优先走 FastLanguageModel(原生2x推理) ──
    # 之前用纯 transformers+PeftModel.generate 实测仅 ~1.5tok/s(76s/条),
    # unsloth for_inference 开启后经 Triton kernel 提速, 预期 30-50tok/s。
    # 失败则降级回 transformers 路径(口径不变, 仅慢)。--no_fast 掉或非cuda时跳过。
    use_fast = use_adapter and device == "cuda" and args.load_in_4bit and not args.no_fast
    if use_fast:
        try:
            from unsloth import FastLanguageModel
            print("  [unsloth FastLanguageModel 路径] 加载基座+adapter (4bit)...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.lora_adapter_dir,
                max_seq_length=2048,        # 评测题干+选项+推理 < 2048, 够用
                dtype=torch.bfloat16,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)   # ← 原生 2x 推理开关
            use_fast = True
        except Exception as e:
            print(f"  ⚠️ unsloth 加载失败({type(e).__name__}: {e}), 降级回 transformers 路径")
            use_fast = False
    if not use_fast:
        if device == "cuda" and args.load_in_4bit:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            if use_adapter:
                # 先 4-bit 加载未量化基座, 再套 LoRA adapter
                from peft import PeftModel
                base = AutoModelForCausalLM.from_pretrained(
                    args.base_model,
                    quantization_config=bnb,
                    low_cpu_mem_usage=True,
                    device_map="cuda",
                )
                model = PeftModel.from_pretrained(base, args.lora_adapter_dir)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_dir,
                    quantization_config=bnb,
                    low_cpu_mem_usage=True,
                    device_map="cuda",
                )
        else:
            if use_adapter:
                from peft import PeftModel
                base = AutoModelForCausalLM.from_pretrained(
                    args.base_model,
                    dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    device_map="cuda" if device == "cuda" else "cpu",
                )
                model = PeftModel.from_pretrained(base, args.lora_adapter_dir)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_dir,
                    dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    device_map="cuda" if device == "cuda" else "cpu",
                )
    model.eval()
    tok_src = args.lora_adapter_dir if use_adapter else args.model_dir
    # unsloth 路径已自带 tokenizer; 走 transformers 路径才单独加载
    if not use_fast:
        if use_adapter:
            try:
                tokenizer = AutoTokenizer.from_pretrained(args.lora_adapter_dir)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    print(f"  ✅ 模型加载完成 ({time.time()-t0:.1f}s)" +
          (f" [基座+LoRA adapter]" if use_adapter else "") +
          f", 参数量 {sum(p.numel() for p in model.parameters())/1e9:.4f}B")
    if device == "cuda":
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  显存: 已占用 {reserved:.2f} GB / {total:.2f} GB "
              f"({reserved/total*100:.0f}%)")
        if reserved / total > 0.7:
            print("  ⚠️ 显存占用已>70%,建议先清其他显存再跑")

    # ── 2. 加载数据 ──
    print(f"\n[2/4] 加载 {DATASET_NAME} (split={DATASET_SPLIT}) ...")
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    n_total = len(ds)
    print(f"  MedQA test: {n_total} 条 (金标字段 answer_idx)")

    # ── 污染自检: 打印前 5 条题干供人工核验 ──
    if args.pollution_print:
        print("\n  ⚠️ 污染自检 —— 请人工扫一眼前 5 条题干, 看有没有明显是训练原题:")
        print("  " + "-" * 60)
        for i in range(min(5, n_total)):
            q = ds[i]["question"]
            ai = ds[i]["answer_idx"]
            opts = ds[i]["options"]
            keys = sorted(opts.keys())
            print(f"  [{i}] 金标={ai}  preds_opt={keys}")
            print(f"      题: {q[:160]}{'...' if len(q) > 160 else ''}")
        print("  " + "-" * 60)
        print("  如以上题干你看着【完全没印象】= 无明显污染; 若【明显是训练原题】请告知。")

    # ── 抽样 ──
    if args.num_mc > 0 and args.num_mc < n_total:
        g = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(n_total, generator=g)[:args.num_mc].tolist()
        print(f"  随机抽样 {len(perm)}/{n_total} 条 (seed={args.seed})")
    else:
        perm = list(range(n_total))
        print(f"  全量评估 {n_total} 条")

    # ── resume ──
    results = []
    done_ids = set()
    resume_file = args.resume_from or out_json
    do_resume = args.resume or args.resume_from
    if do_resume and os.path.exists(resume_file):
        try:
            with open(resume_file, "r", encoding="utf-8") as f:
                prev = json.load(f)
            results = prev.get("results", [])
            done_ids = {r["idx"] for r in results}
            print(f"  🔁 resume: 跳过已完成 {len(done_ids)} 条 (from {resume_file})")
        except Exception as e:
            print(f"  ⚠️ resume 读取失败({e}),从头开始")

    # ── 3. 逐条评估 ──
    print(f"\n[3/4] 逐条贪心推理 + 金标比对 ({len(perm)} 条)...")
    txt_fp = open(out_txt, "a", encoding="utf-8")

    def _flush_json():
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"results": results, "config": vars(args)}, f,
                      ensure_ascii=False, indent=2)

    hit = 0; wrong = 0; no_letter = 0; err = 0
    per_letter = {c: {"hit": 0, "tot": 0} for c in "ABCDE"}
    pred_dist = {c: 0 for c in "ABCDE"}    # 模型预测字母分布(看是否在猜)
    by_source = {"pattern": {"hit": 0, "tot": 0},
                 "fallback": {"hit": 0, "tot": 0}}

    use_batch = args.batch_size and args.batch_size > 1 and device == "cuda"
    pad_id = tokenizer.eos_token_id
    # 未完成条目元数据(idx, question文本, gold, valid), 按 perm 顺序
    todo = []
    for i in perm:
        if i in done_ids:
            continue
        q = ds[i]["question"]; opts = ds[i]["options"]
        todo.append({
            "idx": i, "mc_q": build_mc_question(q, opts),
            "gold_idx": ds[i]["answer_idx"].upper(),
            "valid": set(k.upper() for k in opts.keys()),  # 恒 {A,B,C,D}
            "q": q, "opts": opts,
        })

    def _score_one(t, raw, reasoning, pred_letter, src):
        """对单条结果分桶 + 累计统计 + append 到 results。统一供单条/batch 路径用。"""
        nonlocal hit, wrong, no_letter, err
        gold_idx = t["gold_idx"]
        bucket = None
        if pred_letter is None and src == "error":
            bucket = "eval_error"; err += 1
        elif pred_letter is None:
            bucket = "no_letter"; no_letter += 1
        elif pred_letter == gold_idx:
            bucket = "hit"; hit += 1
            if gold_idx in per_letter:
                per_letter[gold_idx]["hit"] += 1
        else:
            bucket = "wrong"; wrong += 1
        if gold_idx in per_letter:
            per_letter[gold_idx]["tot"] += 1
        if pred_letter is not None:
            pred_dist[pred_letter] = pred_dist.get(pred_letter, 0) + 1
        if src in by_source:
            by_source[src]["tot"] += 1
            if bucket == "hit":
                by_source[src]["hit"] += 1
        results.append({
            "idx": t["idx"], "bucket": bucket,
            "gold_idx": gold_idx, "pred_letter": pred_letter,
            "extract_source": src, "meta_info": ds[t["idx"]].get("meta_info", ""),
            "question": t["q"][:400],
            "options": {k: str(v)[:120] for k, v in t["opts"].items()},
            "model_reasoning": (reasoning or "")[:300],
            "model_answer": (raw)[:500],
        })
        return bucket

    print(f"  ... 评测模式: {'batch=' + str(args.batch_size) if use_batch else '逐条'} {len(todo)} 条")

    if use_batch:
        bs = args.batch_size
        for s in range(0, len(todo), bs):
            batch = todo[s:s+bs]
            mc_qs = [b["mc_q"] for b in batch]
            t_batch0 = time.time()
            try:
                raws = generate_batch(model, tokenizer, mc_qs, device,
                                     args.max_new_tokens, pad_id)
                # 逐条抽字母 + 分桶
                for b, raw in zip(batch, raws):
                    try:
                        reasoning, answer = extract_think_answer(raw)
                        pred_letter, src = extract_answer_letter(answer or raw,
                                                                valid_letters=b["valid"])
                    except Exception:
                        raw = f"<eval_error: {type(Exception().__class__).__name__}>"
                        reasoning, answer = None, None
                        pred_letter, src = None, "error"
                    bucket = _score_one(b, raw, reasoning, pred_letter, src)
                    status = {"hit": "✅命中", "wrong": "❌答错",
                              "no_letter": "￤未给字母", "eval_error": "￤评估失败"}[bucket]
                    print(f"  [idx={b['idx']}] {status} gold={b['gold_idx']} pred={pred_letter} (src={src})")
            except Exception as e:
                # 整批都炸(万一只剩 fallback 路径也救不了), 逐条标 error 不中断
                print(f"  ⚠️ batch 异常 ({type(e).__name__}), 整批标 error")
                for b in batch:
                    _score_one(b, f"<eval_error: {type(e).__name__}>",
                               None, None, "error")
            print(f"  [batch {s//bs+1}/{(len(todo)+bs-1)//bs}] 本批 {len(batch)} 条 用 {time.time()-t_batch0:.1f}s")
            if (s // bs + 1) % 1 == 0:   # 每批都 flush(批少, 容错高)
                _flush_json()
    else:
        # 逐条路径(原口径, batch_size<=1 或 CPU 时)
        for n, t in enumerate(todo):
            try:
                raw = generate_answer(model, tokenizer, t["mc_q"], device,
                                      max_new_tokens=args.max_new_tokens)
                reasoning, answer = extract_think_answer(raw)
                pred_letter, src = extract_answer_letter(answer or raw,
                                                          valid_letters=t["valid"])
            except Exception as e:
                raw = f"<eval_error: {type(e).__name__}>"
                reasoning, answer = None, None
                pred_letter, src = None, "error"
            bucket = _score_one(t, raw, reasoning, pred_letter, src)
            status = {"hit": "✅命中", "wrong": "❌答错",
                      "no_letter": "￤未给字母", "eval_error": "￤评估失败"}[bucket]
            print(f"  [{n+1}/{len(todo)}] idx={t['idx']} {status} "
                  f"gold={t['gold_idx']} pred={pred_letter} (src={src})")
            if (n + 1) % 5 == 0:
                _flush_json()

    _flush_json()
    scored = hit + wrong + no_letter + err
    acc = hit / scored * 100 if scored else 0.0
    # 二项分布 95% 近似区间(随机基线对 4 选 = 25%)
    import math
    z = 1.96
    p = hit / scored if scored else 0.0
    ci = z * math.sqrt(p * (1 - p) / scored) * 100 if scored else 0.0
    baseline = 25.0  # 4 选项均匀随机期望

    print("\n" + "=" * 64)
    print("  评估汇总 (MedQA USMLE 真金标)")
    print("=" * 64)
    print(f"  ⭐ 准确率: 命中 {hit}/{scored} = {acc:.1f}%")
    print(f"     95% 置信区间 ≈ ±{ci:.1f}%  (基线: 4选项随机=25%)")
    print(f"     答错 {wrong} | 未给字母 {no_letter} | 评估失败 {err}")
    print(f"\n  vs 旧口径 (medical-o1 Response 反抽参考): 30.0% ± 14%  → 仅略高于随机")
    if acc - ci > baseline:
        print(f"  ✅ 本批 {acc:.1f}% - {ci:.1f}% = {acc-ci:.1f}% > 25% 随机基线 ⇒ 模型【确有】选择题能力")
    elif acc + ci < baseline:
        print(f"  ❌ 本批 {acc:.1f}% + {ci:.1f}% = {acc+ci:.1f}% < 25% ⇒ 异常(低于随机, 疑抽取/格式问题)")
    else:
        print(f"  ⚠️ 本批与随机基线在统计上无法显著区分(区间与25%重叠)")

    print(f"\n  每个金标选项命中分布:")
    for c in "ABCDE":
        s = per_letter[c]
        if s["tot"]:
            print(f"    {c}: {s['hit']}/{s['tot']} = {s['hit']/s['tot']*100:.0f}%")
        else:
            print(f"    {c}: 0/0")

    print(f"\n  模型预测字母分布 (看是否在猜):")
    print("    " + "  ".join(f"{c}={pred_dist.get(c,0)}" for c in "ABCDE"))
    print(f"  金标分布:")
    gold_dist = {c: per_letter[c]["tot"] for c in "ABCDE"}
    print("    " + "  ".join(f"{c}={gold_dist.get(c,0)}" for c in "ABCDE"))

    print(f"\n  抽字母来源分桶 (噪声对照, 与旧报告同口径):")
    for s, d in by_source.items():
        if d["tot"]:
            print(f"    {s}: {d['hit']}/{d['tot']} = {d['hit']/d['tot']*100:.0f}%")

    print(f"\n  结果文件: {out_json} / {out_txt}")
    print("✅ 评估完成!")

    # 写 txt 人读明细
    txt_fp.write("=" * 64 + "\nMedQA(USMLE) 选择题命中明细 (真金标)\n" + "=" * 64 + "\n")
    for r in results:
        txt_fp.write(f"\nidx={r['idx']} {r['bucket']} gold={r['gold_idx']} "
                     f"pred={r['pred_letter']} (src={r['extract_source']}) "
                     f"meta={r['meta_info']}\n")
        txt_fp.write(f"【问题】{r['question']}\n")
        txt_fp.write(f"【选项】{r['options']}\n")
        txt_fp.write(f"【模型】{r['model_answer']}\n")
        txt_fp.write("-" * 64 + "\n")
    txt_fp.close()


if __name__ == "__main__":
    main()
