import os, sys
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, r"D:\youxi\Trae\Trae_EN\test3")
import torch
from unsloth import FastLanguageModel
import eval_medqa
from datasets import load_dataset
ADAPTER = "output_qwen15b_medical_o1_rank64/checkpoint-2300"
model, tokenizer = FastLanguageModel.from_pretrained(model_name=ADAPTER, max_seq_length=4096, dtype=torch.bfloat16, load_in_4bit=True)
FastLanguageModel.for_inference(model)
ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
for i in [908, 647, 1102]:
    row = ds[i]
    q=row["question"]; opts=row["options"]; gold=row["answer_idx"]
    mc = eval_medqa.build_mc_question(q, opts)
    ids, am = eval_medqa.build_prompt(tokenizer, mc)
    ids=ids.to("cuda"); am=am.to("cuda")
    with torch.inference_mode():
        out = model.generate(input_ids=ids, attention_mask=am, max_new_tokens=900, do_sample=False, pad_token_id=tokenizer.eos_token_id, use_cache=True)
    raw = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    _, answer = eval_medqa.extract_think_answer(raw)
    print(f"\n===== idx={i} gold={gold} =====")
    print(f"选项: {opts}")
    print(f"答案正文 全文:\n{answer if answer else raw}")
    print(f"--- (上面正文提到哪个选项内容? A={opts.get('A')[:40]!r} 等)")
