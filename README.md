# 🏥 Medical Qwen — 医疗问答微调

[![Unsloth](https://img.shields.io/badge/Unsloth-2026.6.9-blue)](https://unsloth.ai)
[![Model](https://img.shields.io/badge/Model-Qwen2.5--1.5B--Instruct--bnb--4bit-green)](https://huggingface.co/unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit)
[![Dataset](https://img.shields.io/badge/Dataset-medical--o1--reasoning--SFT-orange)](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)
[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-1.5B_Model-yellow)](https://huggingface.co/xjh666/medical-o1-qwen2.5-1.5b)
[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-0.5B_Legacy-lightgrey)](https://huggingface.co/xjh666/medical-o1-qwen2.5-0.5b)
[![GitHub](https://img.shields.io/badge/GitHub-Source-lightgrey)](https://github.com/bbpeople/medical-o1-finetune)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

基于 **Qwen2.5-1.5B-Instruct**（在 RTX 3050 4GB 上需 `MAX_SEQ_LENGTH=512`），使用 **Unsloth + QLoRA** 微调医疗推理问答。

> 📦 **当前 1.5B 权重 (rank64)**：[xjh666/medical-o1-qwen2.5-1.5b](https://huggingface.co/xjh666/medical-o1-qwen2.5-1.5b)
> 📦 **历史 0.5B 权重**：[xjh666/medical-o1-qwen2.5-0.5b](https://huggingface.co/xjh666/medical-o1-qwen2.5-0.5b)（已发布，仅作容量对比降级基线）

模型学会先进行 `<think>` 推理链思考，再给出最终答案（类似 o1 风格）。

> 🔧 **评估抽取修复**：fine-tuned 1.5B 在 MedQA 的最终 answer 段常以散文形式 `The most likely X is:

B) Option text` 或 markdown bold `**A) Option text**` 指认答案，无 "answer is" 语义前缀。`eval_medqa.py` 的 `_ANSWER_PATTERNS` 在 9 个语义 pattern 基础上 追加 2 个高约束兜底 pattern（`X) + 大写选项文本`、markdown bold），把 93/100 条从噪声 `fallback` 路径拉进干净 `pattern` 路径，准确率统计不再被裸字 fallback 污染。

---

## 📊 训练结果（1.5B，rank64）

| 指标 | 1.5B / rank64 (1 epoch) |
|------|:-------:|
| 验证 Loss (最终) | **1.539** |
| MedQA(USMLE) 命中率 (100条, t900) | **37%** (口径C) / 33.5% pattern路径 |
| 训练步数 | 2300 (≈1 epoch) |
| 峰值显存 | ~4 GB (RTX 3050 Laptop, 4-bit QLoRA) |

> **评测说明**：MedQA(USMLE-4-options) test 集 100 条随机抽样，max_new_tokens=900，贪心解码。
> 命中率分三口径——A(no_letter算错)=33.0%、B(剔除no_letter)=34.4%、**C(no_letter算对)=37.0%**。
> src 路径分层：`pattern` 干净抽取 93/100 条命中率 35.5%，`fallback` 裸字母 3 条全错(噪声)，`no_letter`(900 token截断) 4 条。
> **对比基线**：随机猜 25%。rank64 checkpoint-2300 实测约高于随机 8-12 个百分点，未显著拉开——1.5B 底座容量为本阶段瓶颈。

---

## 📊 历史 0.5B 基线（已发布，见下方历史仓库）

| 指标 | 1 epoch | 2 epoch |
|------|:-------:|:-------:|
| 训练 Loss | 1.70 | — |
| 验证 Loss (最终) | **1.692** | — |
| 答案关键词重叠 (50条) | **41%** | **48%** |
| 推理关键词重叠 (50条) | **39%** | **39%** |
| 训练耗时 | ~6h | ~12h |
| 峰值显存 | 6.1 GB (merge阶段) | — |

> **基线对比**：未训练的 checkpoint-50 答案重叠仅 18%。2 epoch 提升至 48%，模型在 ICU 感染判断、尿道解剖定位、孕产妇管理等医学问题上均表现正确。**升级动机**：0.5B 容量有限，在 HAP 病原谱、Wernicke 脑病等需要注入临床知识的鉴别诊断上仍答错，故切到 1.5B 对比。

---

## 🚀 快速开始

### 环境要求

| 项目 | 要求 |
|------|------|
| GPU | NVIDIA GPU ≥ 4GB VRAM (RTX 3050 可用) |
| 驱动 | NVIDIA 驱动 ≥ 550（支持 CUDA 12.1+） |
| Python | 3.10 或 3.12 |
| 硬盘 | ≥ 10GB 可用空间 |

### 使用预训练模型（推荐）

训练好的 LoRA 适配器已上传 HuggingFace，几行代码即可加载：

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="xjh666/medical-o1-qwen2.5-1.5b",  # 当前 1.5B rank64 adapter (自动加载 LoRA + 基座)
    # model_name="xjh666/medical-o1-qwen2.5-0.5b",  # 历史 0.5B 基线
    max_seq_length=1024,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

messages = [
    {"role": "user", "content": "What causes fever and productive cough 48h after ICU admission?"},
]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
).to("cuda")

outputs = model.generate(input_ids=inputs, max_new_tokens=512, temperature=0.1)
print(tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True))
```

> 完整 16-bit 合并权重也一并上传，可在 `merged_16bit/` 目录下载。

---

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
python train_medical_o1.py --resume_from_checkpoint ./output_qwen15b_medical_o1/checkpoint-2000

# 中文数据（修改脚本中 DATASET_CONFIG = "zh"）
```

### 训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 基座模型 | Qwen2.5-1.5B-Instruct-bnb-4bit | 4-bit 量化 |
| LoRA rank | 16 | — |
| Batch size | 1 × 8 (gradient accum) | 有效 batch=8 |
| Max seq length | 512 | 适配 4GB 显存（1.5B 需降 seq） |
| Learning rate | 2e-4 | cosine 调度 |
| 优化器 | adamw_8bit | 8-bit 省显存 |
| 保存策略 | 每 200 步 | 自定义 callback，保留最近 3 个 |

### 训练监控

```bash
tensorboard --logdir ./output_qwen15b_medical_o1
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
python eval_medical_o1.py --checkpoint ./output_qwen15b_medical_o1/checkpoint-2000 --num_samples 50
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

## 🩺 MedQA(USMLE) 金标准选择确率评测

旧的 `eval_medical_o1.py` 是"从模型 Response 反抽关键词重叠"，无真金标、有噪声、置信区间宽(±14%)，与随机猜(25%)无法区分。`eval_medqa.py` 换成带真金标的独立测试集 [GBaker/MedQA-USMLE-4-options](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options)，`answer_idx` 直接对错，拿到可信的专业选择题命中。

```bash
# rank64 checkpoint-2300, 100 条随机抽样, max_new_tokens=900, 单条贪心
python -u eval_medqa.py \
  --lora_adapter_dir output_qwen15b_medical_o1_rank64/checkpoint-2300 \
  --base_model unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit \
  --num_mc 100 --max_new_tokens 900 --batch_size 0 \
  --out_prefix eval_medqa_rank64_t900_fixpat

# 断点续跑(--resume --resume_from 已有 JSON, 避免重跑已完成条目)
python -u eval_medqa.py ... --resume --resume_from eval_medqa_rank64_t900_fixpat_<ts>.json
```

**实测 100 条结果（rank64, t900）**：

| bucket | 条数 | 说明 |
|------|:---:|------|
| hit | 33 | 命中 |
| wrong | 63 | 答错 |
| no_letter | 4 | 900 token 截断未给字母 |

- 口径A (no_letter算错) = 33.0%，口径B (剔除no_letter) = 34.4%，**口径C (no_letter算对) = 37.0%**
- `pattern` 干净抽取路径 93/100，命中率 35.5%；`fallback` 裸字母 3 条全错(噪声)；`no_letter` 4 条截断
- 模型存在"偏A/D、避B"的兜底字母倾向(答错题中 pred A 占 37%)，但答错题里 0 条在 answer 段提到正确 gold 字母——是真不会而非"会了没选对"，故字母校准不可救

---

```
├── train_medical_o1.py         # 训练脚本
├── eval_medical_o1.py          # 评估脚本（关键词重叠，旧口径，无真金标）
├── eval_medqa.py               # MedQA(USMLE) 金标准选择确率评测（带真金标）
├── train_medical_o1_rank64.py  # 训练脚本（rank64 变体）
├── requirements.txt            # 依赖清单
├── README.md                   # 本文件
├── .gitignore
├── output_qwen15b_medical_o1/          # 训练输出(rank16, gitignore)
└── output_qwen15b_medical_o1_rank64/  # 训练输出(rank64, gitignore)
    ├── lora_adapter/           # LoRA 适配器
    ├── merged_16bit/           # 完整 16-bit 合并权重
    ├── checkpoint-N/           # 中间 checkpoint（每 100 步，含 -2100/-2200/-2300）
    └── runs/                   # TensorBoard 日志
```

---

## ⚙️ 配置说明

`train_medical_o1.py` 开头的配置段可自由调整：

```python
# 显存优化（4GB VRAM 建议值）
BATCH_SIZE = 1                    # 4GB VRAM 最小值
GRADIENT_ACCUMULATION_STEPS = 8   # 可升到 8 补偿 batch (有效 batch=8)
MAX_SEQ_LENGTH = 512              # 1.5B 在 4GB VRAM 下需 512；0.5B 可用 1024
```

---

## 🐛 常见问题

| 问题 | 解决方法 |
|------|---------|
| `PicklingError: SFTConfig` | 已内置 `ManualSaveCallback` 绕过，保持 `save_strategy="no"` |
| OOM (显存不足) | 降 `MAX_SEQ_LENGTH`、降 `BATCH_SIZE`、升 `GRADIENT_ACCUMULATION_STEPS` |
| `PYTHONPATH` 冲突 | 运行时 `PYTHONPATH="" python train_medical_o1.py` |
| CUDA 不可用 | 检查 `nvidia-smi`，重装匹配的 PyTorch |

### 关于 checkpoint 恢复（resume）

`ManualSaveCallback` 每个 checkpoint 写入以下文件，全部绕过 HF 原生 `_save_checkpoint` 的 pickle 路径：

- `adapter_model.safetensors` + `adapter_config.json` —— LoRA 权重
- `trainer_state.json` —— HF 标准 `TrainerState`，记录 `global_step` / `epoch`
- `optimizer.pt` + `scheduler.pt` —— **尽力保存**；8-bit 优化器（bitsandbytes / paged_adamw）的 state 有时无法被 `torch.save` pickle，失败则自动跳过
- `rng_state.pth` —— 复现随机性，失败仅跳过

恢复时的行为：

| checkpoint 内容 | resume 行为 |
|------|------|
| 含 `trainer_state.json` + `optimizer.pt` + `scheduler.pt` | **完整无偏续训**（权重 + 优化器 + 调度器 + 步数全部恢复）|
| 含 `trainer_state.json`，缺 `optimizer.pt`/`scheduler.pt` | **近似续训**（恢复 LoRA 权重 + 学习率位置，优化器动量重置）|
| 旧式 checkpoint（只有 `training_state.pt`，无 `trainer_state.json`）| 自动回退到「加载 LoRA 权重 + 从 0 训练」（旧行为）|

> ⚠️ **旧 checkpoint 兼容性**：升级到 1.5B 之前、由旧脚本（0.5B 时期）产生的 checkpoint 目录**没有 `trainer_state.json`**，新 resume 逻辑会自动把它们识别为旧式并走"加载权重 + 从 0 训练"路径——**不会**做真正的步数续训。要从 1.5B 训练中途恢复，请用**新脚本产生的 checkpoint**。  
> ⚠️ "近似续训"指 optimizer 动量（Adam 的一阶/二阶矩）被重置，学习率会从断点位置继续走 cosine，但不等于数学上无偏的续训。完整无偏续训要求 optimizer.pt 成功保存。

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
