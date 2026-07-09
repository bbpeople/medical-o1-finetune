"""
rank64 fused CE 崩溃根因探针

目标: 在真实的训练 forward 路径里, 拦截 unsloth_fused_ce_loss 被调用的瞬间,
dump 出三条决定 fused CE 是否触发的关键信息:
  1. 运行时 os.environ["UNSLOTH_RETURN_LOGITS"] 的真实值
  2. 调用 fused CE 时 GPU 空闲显存 (torch.cuda.mem_get_info)
  3. (可选) bsz / q_len 规模

跑完即知: env 到底是不是 "1", 以及是不是显存归零导致的报错。
仅诊断, 不改训练行为。
"""
import os
# 钉死env, 与 train_medical_o1.py 第14行同口径, 但更早
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

import torch
# 必须先 import unsloth(会正确初始化 unsloth_zoo), 再去 patch fused CE 子模块;
# 直接 import unsloth_zoo.* 子模块会触发 zoo __init__ 的安装校验而崩。
import unsloth  # noqa: F401  触发 zoo 初始化
import unsloth_zoo.fused_losses.cross_entropy_loss as ce_mod

_orig_fused_ce = ce_mod.unsloth_fused_ce_loss
_call_count = [0]

def _spying_fused_ce(*args, **kwargs):
    _call_count[0] += 1
    env_val = os.environ.get("UNSLOTH_RETURN_LOGITS", "<unset>")
    try:
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            free_gb = free / 1024**3
            total_gb = total / 1024**3
        else:
            free_gb = total_gb = -1.0
    except Exception as e:
        free_gb = total_gb = f"<err:{e}>"
    # 取 hidden_states 看 bsz/q_len
    hs = kwargs.get("hidden_states")
    shape = tuple(hs.shape) if hs is not None and hasattr(hs, "shape") else "?"
    print(f"\n>>> [ fused CE 被调用 #{_call_count[0]} ] "
          f"UNSLOTH_RETURN_LOGITS={env_val!r}  "
          f"GPU free={free_gb}GB total={total_gb}GB  "
          f"hidden_states.shape={shape}", flush=True)
    # 不放行真实调用(避免又崩), 训练会因缺 loss 报错退出 —— 但我们要的信息已打出
    raise SystemExit("🔍 诊断信息已收集, 主动退出 (未跑真实 fused CE)")

ce_mod.unsloth_fused_ce_loss = _spying_fused_ce
# llama.py 是 from ... import unsloth_fused_ce_loss, 也要拦它的符号
import unsloth.models.llama as llama_mod
llama_mod.unsloth_fused_ce_loss = _spying_fused_ce

import train_medical_o1
train_medical_o1.LORA_R = 64
train_medical_o1.LORA_ALPHA = 64
train_medical_o1.OUTPUT_DIR = "./output_qwen15b_medical_o1_rank64_probe"

if __name__ == "__main__":
    print("=" * 60)
    print("🔍 rank64 fused CE 根因探针 (探完即退, 不真训)")
    print("=" * 60)
    train_medical_o1.main()
