"""
=============================================================================
 独立合并脚本：把已训练好的 LoRA adapter 合并到 16-bit 完整权重
=============================================================================
 背景:
   train_medical_o1.py 训练完成后会调用 Unsloth 的 save_pretrained_merged(merged_16bit)。
   该步需要从 HF 下载未量化的基座权重(~3GB)，在 CPU/内存中反量化 + merge LoRA，
   峰值吃 8~12GB 系统 RAM。本机仅 15.6GB 物理内存，训练刚结束时残留占用导致 OOM，
   python 进程被静默终止 → merged_16bit/ 只剩 tokenizer/config，缺 model.safetensors。

 本脚本:
   - 用 kcsj_new 环境的 transformers 5.5 + peft 0.19 标准路径（不依赖 Unsloth）
   - 强制 CPU、fp16、low_cpu_mem_usage 流式加载未量化基座，峰值 RAM ≈ 6~7GB
   - merge_and_unload() 后分片保存到 merged_16bit/
   - idempotent：若 merged_16bit/model.safetensors 已存在则跳过

 启动前请先确保基座权重已在本地缓存（用 dl_base_weights.py 通过 hf-mirror 镜像下好，
   避免 merge 时再触发网络下载而卡死/OOM）。

 用法:
   D:\\anaconda\\envs\\kcsj_new\\python.exe merge_medical_o1.py
=============================================================================
"""

import os
import gc
import shutil
import sys
import time

# 优先用国内镜像下载元数据/补漏小文件；大权重已在缓存里时不会触发下载。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ── 配置 ──────────────────────────────────────────────────────────────────
# 基座必须用【未量化】版，否则 merge 会失真。adapter_config 里写的是 bnb-4bit 量化版，
# 这里硬覆盖为同名的未量化 unsloth 版（Unsloth 内部 merge 也是这么做的）。
#
# 优先用本地目录 base_local/unsloth-Qwen2.5-1.5B-Instruct（由 dl 小脚本 + curl 从
# hf-mirror 拉好），避免 merge 时再走 HF 的 xet/endpoint 下载而卡死/OOM。
_LOCAL_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "base_local", "unsloth-Qwen2.5-1.5B-Instruct")
# 本地基座 model.safetensors 必须完整（>2GB 才认，否则可能 curl 下了一半）
_local_sft = os.path.join(_LOCAL_BASE, "model.safetensors")
if os.path.exists(_local_sft) and os.path.getsize(_local_sft) > 2 * 1024**3:
    BASE_MODEL = _LOCAL_BASE
else:
    BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct"   # 回退到 HF（需网络/镜像）
ADAPTER_DIR = "output_qwen15b_medical_o1/lora_adapter"
OUTPUT_DIR = "output_qwen15b_medical_o1/merged_16bit"
SHARD_SIZE = "500MB"        # 分片落盘，降低序列化峰值


def _mem(cap):
    return (cap.total / 1024**3, cap.available / 1024**3) if cap else (float("nan"), float("nan"))


def _log_mem(tag):
    import psutil
    tot, avail = _mem(psutil.virtual_memory()) if psutil else (0, 0)
    rss = 0
    try:
        rss = psutil.Process(os.getpid()).memory_info().rss / 1024**3 if psutil else 0
    except Exception:
        pass
    print(f"  [mem:{tag}] total={tot:.1f}GB avail={avail:.1f}GB process RSS={rss:.2f}GB")


def main():
    # idempotent 检查
    merged_ok = (os.path.isdir(OUTPUT_DIR)
                 and any(f.endswith(".safetensors") for f in os.listdir(OUTPUT_DIR)))
    if merged_ok:
        print(f"✅ 已检测到完整合并产物（{OUTPUT_DIR} 下有 *.safetensors），跳过。")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        import psutil
    except Exception:
        psutil = None
    print(f"kcsj_new 合并脚本启动 at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    _log_mem("startup")

    # ── 1. 加载 tokenizer（直接从 adapter 目录复制即可，已在训练时保存）──
    print("\n[1/4] 准备 tokenizer...")
    tok_src = ADAPTER_DIR
    # 直接复用训练时已存的 tokenizer，避免重复下载/版本漂移
    if os.path.exists(os.path.join(tok_src, "tokenizer.json")):
        tokenizer = AutoTokenizer.from_pretrained(tok_src)
    else:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    # 保存到输出目录（覆盖掉 merge 中断时残留的半个目录里的同名文件）
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("  ✅ tokenizer 已保存")

    # ── 2. 流式加载未量化基座到 CPU（峰值低）──
    print(f"\n[2/4] 加载未量化基座 {BASE_MODEL} (CPU, fp16, low_cpu_mem_usage)...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,        # 流式加载，避免一次性全量驻留
        device_map="cpu",              # 强制 CPU，不碰显存
    )
    base.eval()
    _log_mem("after base load")

    # ── 3. 套用 LoRA 并 merge ──
    print(f"\n[3/4] 加载 LoRA adapter: {ADAPTER_DIR} → merge_and_unload() ...")
    peft_model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    _log_mem("after peft load")
    merged = peft_model.merge_and_unload()
    _log_mem("after merge_and_unload")

    # 释放中间对象，给落盘腾内存
    del base, peft_model
    gc.collect()

    # ── 4. 分片保存 ──
    print(f"\n[4/4] 保存完整 16-bit 权重到 {OUTPUT_DIR} (shard={SHARD_SIZE}) ...")
    # merged_16bit 目录里可能残留 merge 中断时的半个配置，先清掉非 tokenizer 的旧文件
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith(".safetensors") or f in ("config.json", "generation_config.json"):
            continue
        # tokenizer 文件保留
    merged.save_pretrained(
        OUTPUT_DIR,
        safe_serialization=True,
        max_shard_size=SHARD_SIZE,
    )
    # 重新保存 tokenizer 以确保 chat_template.jinja 等齐全
    tokenizer.save_pretrained(OUTPUT_DIR)
    _log_mem("after save")

    print("\n✅ 合并完成！产物在:", OUTPUT_DIR)
    print("  ", [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".safetensors")])


if __name__ == "__main__":
    main()
