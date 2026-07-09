"""probe6: 精确定位抽取失败的根因。
对 probe5 那 5 条(t900 评测里被判 wrong/fallback 的 idx), 打印:
1. extract_think_answer 抽到的 answer 段全文(看模型到底输出啥格式)
2. 9 个 _ANSWER_PATTERNS 逐个 search 的命中情况(看哪个 miss 了)
3. fallback 抓到啥字母
这样加 patterns 才有的放矢, 不盲猜格式。"""
import os, sys
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, r"D:\youxi\Trae\Trae_EN\test3")
import re, time, torch
from unsloth import FastLanguageModel
import eval_medqa
from datasets import load_dataset

ADAPTER = "output_qwen15b_medical_o1_rank64/checkpoint-2300"
print("[probe6] load...", flush=True)
t0 = time.time()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER, max_seq_length=4096, dtype=torch.bfloat16, load_in_4bit=True)
FastLanguageModel.for_inference(model)
print(f"[load] {time.time()-t0:.1f}s", flush=True)

ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
VALID = set("ABCD")

for i in [908, 660, 1102, 647, 904]:
    row = ds[i]
    q = row["question"]; opts = row["options"]; gold = row["answer_idx"]
    mc = eval_medqa.build_mc_question(q, opts)
    ids, am = eval_medqa.build_prompt(tokenizer, mc)
    ids = ids.to("cuda"); am = am.to("cuda")
    FastLanguageModel.for_inference(model)
    with torch.inference_mode():
        out = model.generate(input_ids=ids, attention_mask=am, max_new_tokens=900,
                             do_sample=False, pad_token_id=tokenizer.eos_token_id, use_cache=True)
    raw = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tags=True) \
        if hasattr(tokenizer, "skip_special_tags") else tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    # 复刻 eval_medqa 的真实抽取链路
    reasoning, answer = eval_medqa.extract_think_answer(raw)
    target = answer or raw  # eval 里就是 extract_answer_letter(answer or raw, ...)

    print(f"\n{'='*70}", flush=True)
    print(f"idx={i}  gold={gold}", flush=True)
    print(f"  extract_think_answer -> reasoning={'有(len='+str(len(reasoning))+')' if reasoning else 'None'}, "
          f"answer={'有(len='+str(len(answer))+')' if answer else 'None'}", flush=True)
    print(f"  target(answer or raw) 用的是: {'answer' if answer else 'raw(fallback)'}", flush=True)
    print(f"  ---- target 全文(前600) ----", flush=True)
    print(f"  ```\n{target[:600]}\n```", flush=True)

    # 逐 pattern 命中诊断
    print(f"  ---- 9 个 _ANSWER_PATTERNS 命中诊断 ----", flush=True)
    any_hit = False
    for pi, pat in enumerate(eval_medqa._ANSWER_PATTERNS):
        m = pat.search(target)
        if m:
            letter = m.group(1).upper()
            in_range = letter in VALID
            print(f"    pat{pi} {pat.pattern[:55]:55} -> 命中 letter={letter} in_range={in_range}", flush=True)
            any_hit = True
        # else 不打, 太吵
    if not any_hit:
        print(f"    [全部 miss] 9 个 pattern 无一命中 target", flush=True)

    # fallback 会抓到啥
    fb_matches = list(eval_medqa._FALLBACK_LETTER_RE.finditer(target))
    fb = next((m.group(1).upper() for m in fb_matches if m.group(1).upper() in VALID), None)
    print(f"  fallback: 在 target 里找到 {len(fb_matches)} 个 A-E, 第一个在ABCD内的是 '{fb}'", flush=True)

    # 最终抽取(复刻 eval)
    pred, src = eval_medqa.extract_answer_letter(target, VALID)
    print(f"  >> 最终: pred={pred} src={src}  | 正确? {pred == gold if pred else 'N/A'}", flush=True)
    print(f"  target 末尾 150: ...{target[-150:]}", flush=True)
