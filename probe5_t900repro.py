"""probe5: 复现 t900 评测里 idx=908(被记为 wrong/fallback) 的题, 900 token 看
extract_think_answer 是否真没闭合标签 -> 这决定 5 条全 fallback 是模型没收尾还是抽取bug。"""
import os, sys
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, r"D:\youxi\Trae\Trae_EN\test3")
import time, torch
from unsloth import FastLanguageModel
import eval_medqa
from datasets import load_dataset

ADAPTER = "output_qwen15b_medical_o1_rank64/checkpoint-2300"
print("[probe5] load...", flush=True)
t0=time.time()
model, tokenizer = FastLanguageModel.from_pretrained(model_name=ADAPTER, max_seq_length=4096, dtype=torch.bfloat16, load_in_4bit=True)
FastLanguageModel.for_inference(model)
print(f"[load] {time.time()-t0:.1f}s", flush=True)

ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
# t900 评测那 5 条 idx: 908 660 1102 647 904
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
    raw = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    n_tok = out.shape[1] - ids.shape[1]
    has_close = "" in raw
    reasoning, answer = eval_medqa.extract_think_answer(raw)
    pred, src = eval_medqa.extract_answer_letter(answer or raw, set("ABCD"))
    # 对照: t900 当时的 bucket
    print(f"\nidx={i} gold={gold} 生成{n_tok}tok | 含={has_close} ", flush=True)
    print(f"  extract_think_answer: reasoning={'有' if reasoning else 'None'} answer={'有' if answer else 'None'}", flush=True)
    print(f"  最终抽取: pred={pred} src={src} (answer or raw 走的 {'answer' if answer else 'raw-fallback'})", flush=True)
    print(f"  raw末尾120: ...{raw[-120:]}", flush=True)
