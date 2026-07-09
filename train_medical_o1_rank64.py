"""
=============================================================================
 rank64 变体训练: 1.5B + LoRA rank 64 (alpha 64, 缩放保持 1:1)
=============================================================================
 复用 train_medical_o1.py 全部训练逻辑, 仅覆盖三个常量:
   LORA_R      16 -> 64   (提高可训练容量上限, 实测算 1.5B +0.3GB 显存, 4GB 安全)
   LORA_ALPHA  16 -> 64   (同步提升, 保持缩放 alpha/r=1.0 与原训练口径一致;
                          若只升 rank 不升 alpha, 缩放会掉到 16/64=0.25, LoRA
                          更新幅度变小, 提分效果被削弱)
   OUTPUT_DIR   -> output_qwen15b_medical_o1_rank64   (新目录, 不覆盖现有 1.5B 产物)

 其余参数(MAX_SEQ_LENGTH=512, NUM_EPOCHS=3, LR=2e-4, 8bit optim 等)全部不动。
"控制变量"比法: 唯一变量是可训练容量更多(rank16 -> rank64), 准确率差只能归因于此。
实测 baseline: eval_medqa.py 200条真金标 → rank16版 36.5%, 此版训完重跑同 200
条对比, 看提分。

 ── rank64 踩坑修正(4GB 卡独有) ──────────────────────────────────────────
 rank16 旧训练能跑通, rank64 却在第一个 batch 崩:
   RuntimeError: Unsloth: No or negligible GPU memory available for fused
   cross entropy.  (unsloth_zoo/fused_losses/cross_entropy_loss.py)

 根因: rank64 多占 ~0.3GB 显存后, Unsloth 的 fused CE 在 forward 里调
   torch.cuda.mem_get_info(0) 查 [当前空闲显存], 再按 free_gb * 0.5 算 chunk
   预算(target_gb)。rank64 下空闲显存被压到接近 0, target_gb <= 1e-9 直接抛错。
   源码见 _get_chunk_multiplier():
     if target_gb <= 1e-9: raise RuntimeError("No or negligible ...")

 修法(双保险, 任一生效即可跑通):
   1) UNSLOTH_RETURN_LOGITS=1  早设 —— 让 forward 走标准 logits 路径,
      彻底跳过 fused CE(llama.py:1452 的 else 分支)。给 env 留 chance。
   2) Monkeypatch unsloth_fused_ce_loss —— 把 target_gb=None 强制改成
      一个固定小预算(0.12 = 120MB), 让 chunk CE 用固定工作区、
      永不再 poll mem_get_info / 再因空闲显存归零而崩。
      这是真正的兜底: 即使 env 因 Unsloth 编译缓存/路径问题没生效,
      仍走 chunk CE(自身省显存) 而不报错。

 实跑前预估(标准 logits 路径显存账):
   4bit 模型 ~1GB + rank64 LoRA ~0.3GB + KV(512) ~0.5GB + 激活 ~0.5GB
   + 8bit 优化器 ~0.3GB + 完整 logits 物化 ~0.31GB(512×151936×4B float)
   ≈ 2.9GB, 4GB 卡剩 ~1.1GB 余量, 安全。
   若 env 没生效走兜底 chunk CE, 反而更省(logits 不物化), 余量更大。

 用法(与 train_medical_o1.py 完全一致):
   D:\\anaconda\\envs\\kcsj_new\\python.exe train_medical_o1_rank64.py
   ... --num_epochs 1   (按当前决策跑 1 epoch, 8~12h, 显存压力下先验收 loss 下降)
   ... --resume_from_checkpoint output_qwen15b_medical_o1_rank64/checkpoint-XXX
   首个 checkpoint 在 step 100(~24min), 中途被 kill 可从 checkpoint 续跑。
=============================================================================
"""

import os
# ── 保险 1: 钉死 env, 早于一切 unsloth import ────────────────────────────
# 跳过 fused CE 走标准 logits 路径(4GB 卡余量够物化完整 logits)。
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

import torch
# 先 import unsloth 让 unsloth_zoo 完成初始化(直接 import 子模块会触发安装校验崩)
import unsloth  # noqa: F401
import unsloth_zoo.fused_losses.cross_entropy_loss as _ce_mod
import unsloth.models.llama as _llama_mod


# ── 保险 2: Monkeypatch 兜底, 防 fused CE 因空闲显存归零而崩 ──────────────
# 把 unsloth_fused_ce_loss 的 target_gb=None(运行时 poll 空闲显存)改成固定
# 0.12GB 预算。chunk CE 拿固定工作区, 不再 mem_get_info, 不再报
# "No or negligible GPU memory"。即使保险1的 env 没生效, 也能安全跑通。
_orig_fused_ce = _ce_mod.unsloth_fused_ce_loss

def _patched_fused_ce(*args, **kwargs):
    # unsloth_fused_ce_loss 的签名是位置+关键字混合; llama.py 调用时全用关键字
    # (hidden_states=, lm_head_weight=, labels=, ... target_gb=None)。若 target_gb
    # 缺省或为 None, 强制塞 0.12; 若调用方已显式给了正值, 尊重它。
    tgt = kwargs.get("target_gb", None)
    if tgt is None:
        kwargs["target_gb"] = 0.12
    return _orig_fused_ce(*args, **kwargs)

_ce_mod.unsloth_fused_ce_loss = _patched_fused_ce
# llama.py 是 `from ... import unsloth_fused_ce_loss`, 要同时换掉它持有的符号
_llama_mod.unsloth_fused_ce_loss = _patched_fused_ce


# 复用现有训练脚本的全部逻辑(不复制 300 行, 维护性更好)
import train_medical_o1

# ── 仅覆盖三个常量: rank/alpha 升 64, 输出换新目录 ──────────────────────
train_medical_o1.LORA_R = 64
train_medical_o1.LORA_ALPHA = 64
train_medical_o1.OUTPUT_DIR = "./output_qwen15b_medical_o1_rank64"
# rank64 显存更紧, 训练易被系统休眠/电源等因素中断, 故把首个 checkpoint 提前到
# 100 步(~24min)、验证提前到 50 步, 这样即使中途被 kill 也有 checkpoint 可续跑,
# 不必每次从头。原值 SAVE_STEPS=200 / EVAL_STEPS=100 对 rank16 够用, rank64 加严。
train_medical_o1.SAVE_STEPS = 100
train_medical_o1.EVAL_STEPS = 50

# 也同步打印头里 BASE_SIZE 标注, 让训练日志清楚显示这是 rank64 变体
# (BASE_SIZE 仍为 1.5B, 不动基座)
if __name__ == "__main__":
    print("=" * 60)
    print("  ⚠️ rank64 变体: LoRA rank 16->64, alpha 16->64 (缩放保持 1:1)")
    print(f"  OUTPUT_DIR = {train_medical_o1.OUTPUT_DIR}")
    print("  已装双保险: UNSLOTH_RETURN_LOGITS=1 + fused CE 固定 target_gb=0.12")
    print("  抗中断: SAVE_STEPS=100 / EVAL_STEPS=50 (首ckpt~24min, 可resume续跑)")
    print("  其余参数同 train_medical_o1.py, 唯一变量 = 可训练容量")
    print("=" * 60)
    train_medical_o1.main()
