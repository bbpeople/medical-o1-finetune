"""把 unsloth/Qwen2.5-1.5B-Instruct 的 model.safetensors 通镜像稳下载到 HF 缓存。
带重试 + 断点续传(hf_hub_download 自带)。endpoint 显式逐次传入,避免环境变量未透传。"""
import os, time
from huggingface_hub import hf_hub_download

ENDPOINT = "https://hf-mirror.com"
REPO = "unsloth/Qwen2.5-1.5B-Instruct"
FILES = ["model.safetensors"]   # 其余小文件已在缓存里(config/tokenizer 之前都成功下过)

for fn in FILES:
    for attempt in range(1, 9):
        try:
            print(f"[{fn}] attempt {attempt} ...", flush=True)
            p = hf_hub_download(repo_id=REPO, filename=fn, endpoint=ENDPOINT,
                                resume_download=True)
            print(f"[{fn}] OK -> {p}", flush=True)
            break
        except Exception as e:
            print(f"[{fn}] attempt {attempt} FAIL: {type(e).__name__} {repr(e)[:200]}", flush=True)
            time.sleep(min(2 ** attempt, 30))
    else:
        print(f"[{fn}] 全部重试失败,请检查网络/代理", flush=True)
        raise SystemExit(1)
print("ALL_DONE", flush=True)
