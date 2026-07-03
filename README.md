# 🏥 Medical Qwen — 医疗问答微调

[![Unsloth](https://img.shields.io/badge/Unsloth-2026.6.9-blue)](https://unsloth.ai)
[![Model](https://img.shields.io/badge/Model-Qwen2.5--0.5B--Instruct-bnb--4bit-green)](https://huggingface.co/unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit)
[![Dataset](https://img.shields.io/badge/Dataset-medical--o1--reasoning--SFT-orange)](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

基于 **Qwen2.5-0.5B-Instruct**，使用 **Unsloth + QLoRA** 在 **RTX 3050 (4GB VRAM)** 上微调医疗推理问答。

模型学会先进行 `<think>` 推理链思考，再给出最终答案（类似 o1 风格）。

---

## 📊 训练结果

| 指标 | 1 epoch | 2 epoch |
|------|:-------:|:-------:|
| 训练 Loss | 1.70 | — |
| 验证 Loss (最终) | **1.692** | — |
| 答案关键词重叠 (50条) | **41%** | **48%** |
| 推理关键词重叠 (50条) | **39%** | **39%** |
| 训练耗时 | ~6h | ~12h |
| 峰值显存 | 6.1 GB (merge阶段) | — |

> **基线对比**：未训练的 checkpoint-50 答案重叠仅 18%。2 epoch 提升至 48%，模型在 ICU 感染判断、尿道解剖定位、孕产妇管理等医学问题上均表现正确。

---

## 🚀 快速开始

### 环境要求

| 项目 | 要求 |
|------|------|
| GPU | NVIDIA GPU ≥ 4GB VRAM (RTX 3050 可用) |
| 驱动 | NVIDIA 驱动 ≥ 550（支持 CUDA 12.1+） |
| Python | 3.10 或 3.12 |
| 硬盘 | ≥ 10GB 可用空间 |

### 安装

```bash
# 1. 创建环境
conda create --name unsloth_env python=3.12 -y
conda activate unsloth_env

# 2. 安装 PyTorch (CUDA ≥ 13.0)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# 3. 安装 Unsloth 和项目依赖
pip install unsloth
pip install -r requirements.txt
```

> **Windows 注意**：
> - `bitsandbytes` 需要 Visual C++ Redistributable
> - 运行脚本时请设置 `PYTHONPATH=""` 避免路径冲突
> - `dataloader_num_workers=0`（已在内置代码中配置）

---

## 🎯 训练

```bash
# 1 epoch（快速验证，~6h）
python train_medical_o1.py --num_epochs 1

# 指定步数
python train_medical_o1.py --max_steps 200

# 从 checkpoint 恢复
python train_medical_o1.py --resume_from_checkpoint ./output_qwen05b_medical_o1/checkpoint-2000

# 中文数据（修改脚本中 DATASET_CONFIG = "zh"）
```

### 训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 基座模型 | Qwen2.5-0.5B-Instruct-bnb-4bit | 4-bit 量化 |
| LoRA rank | 16 | — |
| Batch size | 1 × 8 (gradient accum) | 有效 batch=8 |
| Max seq length | 1024 | 适配 4GB 显存 |
| Learning rate | 2e-4 | cosine 调度 |
| 优化器 | adamw_8bit | 8-bit 省显存 |
| 保存策略 | 每 200 步 | 自定义 callback，保留最近 3 个 |

### 训练监控

```bash
tensorboard --logdir ./output_qwen05b_medical_o1
# 浏览器打开 http://localhost:6006
```

---

## 🔍 评估

```bash
# 快速评估（5 条）
python eval_medical_o1.py

# 完整评估
python eval_medical_o1.py --num_samples 50 --output eval_results.txt

# 评估指定 checkpoint
python eval_medical_o1.py --checkpoint ./output_qwen05b_medical_o1/checkpoint-2000 --num_samples 50
```

### 评估输出示例

```
  📝 问题: A 28-year-old G1P0 woman who is 30 weeks pregnant...
  🤖 模型生成:
     🧠 推理链: <think>...推理过程...</think>
     💡 答案: Based on the information provided, the most appropriate
             next step is to perform a biophysical profile (BPP).
  ✅ 标准:
     💡 答案: The best next step in management is D. Biophysical profile.
  📊 关键词重叠 (答案): 43%
```

---

## 🗂️ 项目结构

```
├── train_medical_o1.py         # 训练脚本
├── eval_medical_o1.py          # 评估脚本
├── requirements.txt            # 依赖清单
├── README.md                   # 本文件
├── .gitignore
└── output_qwen05b_medical_o1/  # 训练输出（gitignore）
    ├── lora_adapter/           # LoRA 适配器 (~45MB)
    ├── merged_16bit/           # 完整 16-bit 权重 (~950MB)
    ├── checkpoint-N/           # 中间 checkpoint（每 200 步）
    └── runs/                   # TensorBoard 日志
```

---

## ⚙️ 配置说明

`train_medical_o1.py` 开头的配置段可自由调整：

```python
# 显存优化（4GB VRAM 建议值）
BATCH_SIZE = 1                    # 可降为 1
GRADIENT_ACCUMULATION_STEPS = 8   # 可升到 8 补偿 batch
MAX_SEQ_LENGTH = 1024             # 可降到 512
```

---

## 🐛 常见问题

| 问题 | 解决方法 |
|------|---------|
| `PicklingError: SFTConfig` | 已内置 `ManualSaveCallback` 绕过，保持 `save_strategy="no"` |
| OOM (显存不足) | 降 `MAX_SEQ_LENGTH`、降 `BATCH_SIZE`、升 `GRADIENT_ACCUMULATION_STEPS` |
| `PYTHONPATH` 冲突 | 运行时 `PYTHONPATH="" python train_medical_o1.py` |
| CUDA 不可用 | 检查 `nvidia-smi`，重装匹配的 PyTorch |

---

## 📖 引用

```bibtex
@misc{chen2024huatuogpto1medicalcomplexreasoning,
  title={HuatuoGPT-o1, Towards Medical Complex Reasoning with LLMs},
  author={Junying Chen and Zhenyang Cai and Ke Ji and Xidong Wang and Wanlong Liu and Rongsheng Wang and Jianye Hou and Benyou Wang},
  year={2024},
  eprint={2412.18925},
  archivePrefix={arXiv},
}
```

---

## 📜 License

MIT
