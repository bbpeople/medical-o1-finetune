"""probe4: 量 rank64 模型生成多少 token 才出闭合 think 标签 + 答案。
max_new_tokens 给足 1800, 看 raw 有没有 、在哪、之后是否选了字母。
这决定后续批量评测的合理 max_new_tokens 上限。"""
import os, sys
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, r"D:\youxi\Trae\Trae_EN\test3")
import time, torch
from unsloth import FastLanguageModel
import eval_medqa

ADAPTER = "output_qwen15b_medical_o1_rank64/checkpoint-2300"
print("[probe4] load...", flush=True)
t0=time.time()
model, tokenizer = FastLanguageModel.from_pretrained(model_name=ADAPTER, max_seq_length=4096, dtype=torch.bfloat16, load_in_4bit=True)
FastLanguageModel.for_inference(model)
print(f"[load] {time.time()-t0:.1f}s", flush=True)

qs = [
 ("A 35-year-old man comes to the physician because of itchy, watery eyes for the past week. He has also been sneezing multiple times a day.",
  {"A":"Allergic rhinitis","B":"Viral conjunctivitis","C":"Bacterial conjunctivitis","D":"Dry eye syndrome"}),
 ("A 39-year-old woman is brought to the emergency department because of fevers, chills, and left lower quadrant pain. Her temperature is 39.1C.",
  {"A":"Appendicitis","B":"Diverticulitis","C":"Ovarian torsion","D":"Ectopic pregnancy"}),
]

MNT = 1800
for qi,(q,opts) in enumerate(qs):
    mc = eval_medqa.build_mc_question(q, opts)
    ids, am = eval_medqa.build_prompt(tokenizer, mc)
    ids = ids.to("cuda"); am = am.to("cuda")
    FastLanguageModel.for_inference(model)
    print(f"\n===== 题{qi} (max_new_tokens={MNT}) =====", flush=True)
    t1=time.time()
    with torch.inference_mode():
        out = model.generate(input_ids=ids, attention_mask=am, max_new_tokens=MNT,
                            do_sample=False, pad_token_id=tokenizer.eos_token_id, use_cache=True)
    dt=time.time()-t1
    raw = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    n_tok = out.shape[1] - ids.shape[1]
    has_close = "" in raw
    close_pos = raw.find("") if has_close else -1
    print(f"生成 {n_tok} tok in {dt:.1f}s = {n_tok/dt:.1f}tok/s", flush=True)
    print(f"raw 长度 {len(raw)} 字符", flush=True)
    print(f'含  闭合标签: {has_close}  (位置 close_pos={close_pos}, 即第 {close_pos//4 if has_close else -1} tok 附近)', flush=True)
    reasoning, answer = eval_medqa.extract_think_answer(raw)
    print(f"extract_think_answer: reasoning_len={len(reasoning) if reasoning else 0}, answer_len={len(answer) if answer else 0}", flush=True)
    if answer:
        pred, src = eval_medqa.extract_answer_letter(answer, set("ABCD"))
        print(f"  answer 抽取: pred={pred} src={src}", flush=True)
        print(f"  answer 全文(前300): {answer[:300]}", flush=True)
    else:
        print(f"  ⚠️ 无闭合 , answer=None, 整段 raw 当 answer 去 fallback -> 会乱抓字母", flush=True)
        pred, src = eval_medqa.extract_answer_letter(raw, set("ABCD"))
        print(f"  fallback 抽: pred={pred} src={src} (噪声, 思考链里随便挴的字母)", flush=True)
    print(f"  raw 末尾 200: ...{raw[-200:]}", flush=True)
