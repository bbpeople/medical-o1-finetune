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
"控制变量"比法: 唯一变量是少阶参数更多(rank16 -> rank64), 准确率差只能归因于此。
实测 baseline: eval_medqa.py 200条真金标 → rank16版 36.5%, 此版训完重跑同 200
条对比, 看提分。

 实跑前预估:
   - 现有 rank16 训练峰值显示存 3.186GB(占4GB的80%), 余~0.8GB
   - rank64 增量约 0.3GB(adapter权重 + 8bit优化器状态梯度), 应在 3.5GB 左右
   - 若遇超长样本触发 OOM, 现有 ManualSaveCallback 仍会按 SAVE_STEPS 存 checkpoint;
     可用 --resume_from_checkpoint 续跑; 极端 OOM 需退回 rank16(用现 train_medical_o1.py)

 用法(与 train_medical_o1.py 完全一致):
   D:\\anaconda\\envs\\kcsj_new\\python.exe train_medical_o1_rank64.py
   ... --num_epochs 1   (按当前决策跑 1 epoch, 8~12h, 显存压力下先验收 loss 下降)
   ... --resume_from_checkpoint output_qwen15b_medical_o1_rank64/checkpoint-XXX
=============================================================================
"""

# 复用现有训练脚本的全部逻辑(不复制 300 行, 维护性更好)
import train_medical_o1

# ── 仅覆盖三个常量: rank/alpha 升 64, 输出换新目录 ──────────────────────
train_medical_o1.LORA_R = 64
train_medical_o1.LORA_ALPHA = 64
train_medical_o1.OUTPUT_DIR = "./output_qwen15b_medical_o1_rank64"

# 也同步打印头里 BASE_SIZE 标注, 让训练日志清楚显示这是 rank64 变体
# (BASE_SIZE 仍为 1.5B, 不动基座)
if __name__ == "__main__":
    print("=" * 60)
    print("  ⚠️ rank64 变体: LoRA rank 16->64, alpha 16->64 (缩放保持 1:1)")
    print(f"  OUTPUT_DIR = {train_medical_o1.OUTPUT_DIR}")
    print("  其余参数同 train_medical_o1.py, 唯一变量 = 可训练容量")
    print("=" * 60)
    train_medical_o1.main()
