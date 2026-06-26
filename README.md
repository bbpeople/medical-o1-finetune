# 🏥 Medical Qwen — 医疗问答微调

基于 **Qwen2.5-0.5B-Instruct**，使用 **Unsloth + QLoRA** 在 **RTX 3050 (4GB)** 上微调医疗问答数据。

---

## 目录

- [环境搭建（详细步骤）](#环境搭建详细步骤)
- [训练](#训练)
- [评估](#评估)
- [推理示例](#推理示例)
- [输出文件说明](#输出文件说明)
- [显存优化](#显存优化)
- [引用](#引用)

---

## 环境搭建（详细步骤）

以下每一步都附带 **验证命令**，确保走到下一步前环境是正确的。

### 环境要求

| 项目 | 要求 |
|------|------|
| GPU | NVIDIA RTX 3050 (4GB VRAM) 或同等及以上 |
| 驱动 | NVIDIA 驱动 ≥ 550（支持 CUDA 12.1+） |
| Python | 3.10 或 3.12 |
| 硬盘 | ≥ 10GB 可用空间 |

### Step 1: 创建 conda 环境

打开 **Git Bash** 或 **Anaconda Prompt**，执行：

```bash
# 创建独立环境（已有 Anaconda 的话不需要重装）
conda create --name unsloth_env python=3.12 -y
```

**预期输出**：下载包并提示 `done`。

```
# 激活环境（每次新开终端都要先执行这步）
conda activate unsloth_env
```

**验证**：命令行前面会出现 `(unsloth_env)` 字样。

### Step 2: 确认显卡驱动正常

```bash
nvidia-smi
```

**预期输出**（类似下面的内容）：

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 610.47                 Driver Version: 610.47       CUDA Version: 13.3      |
|-----------------------------------+----------------------+----------------------+
| GPU  Name                  TCC   | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                   |                      |               MIG M. |
|===================================+======================+======================|
|   0  NVIDIA GeForce RTX 3050 ...  |   00000000:01:00.0  |      N/A |
|   N/A   55C    P0    N/A /  N/A  |    500MiB /  4096MiB |     10%      Default |
+-----------------------------------+----------------------+----------------------+
```

> ⚠️ 如果没有输出，说明 NVIDIA 驱动没装好，先装驱动。

### Step 3: 安装 PyTorch（带 CUDA）

```bash
# 必须确保在 unsloth_env 环境下
conda activate unsloth_env

# 安装 PyTorch（cu130 = CUDA 13.0，兼容 13.x 驱动）
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

**预期耗时**：1-5 分钟（约 2GB 下载）。

### Step 4: 验证 PyTorch 能识别 GPU

```bash
python -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用:    {torch.cuda.is_available()}')
print(f'GPU 名称:     {torch.cuda.get_device_name(0)}')
print(f'显存总量:     {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
print(f'BF16 支持:    {torch.cuda.is_bf16_supported()}')
"
```

**预期输出**：

```
PyTorch 版本: 2.10.0+cu130
CUDA 可用:    True
GPU 名称:     NVIDIA GeForce RTX 3050 Laptop GPU
显存总量:     4.00 GB
BF16 支持:    True
```

> ❌ 如果 `CUDA 可用: False`，最常见原因是：
> - 忘记 `conda activate unsloth_env`，装到了其他环境
> - PyTorch 装成了 CPU 版本（检查版本号有没有 `+cu` 后缀）
>
> 解决：重新激活环境，重新执行 Step 3。

### Step 5: 安装 Unsloth

```bash
# 继续在 unsloth_env 环境里
pip install unsloth
```

**预期耗时**：2-5 分钟。Unsloth 会自动安装依赖的 triton、xformers 等。

**验证安装**：

```bash
python -c "
from unsloth import FastLanguageModel
print('Unsloth 导入成功 ✅')
from unsloth import is_bfloat16_supported
print(f'BF16 支持: {is_bfloat16_supported()}')
"
```

> ❌ 如果报 `xformers` 相关错误，尝试：
> ```bash
> conda install xformers -c xformers
> pip install unsloth
> ```

### Step 6: 安装项目依赖

```bash
# 进入项目目录
cd D:/youxi/Trae/Trae_EN/test3

# 安装项目依赖
pip install -r requirements.txt
```

### Step 7: 验证全部安装成功

```bash
cd D:/youxi/Trae/Trae_EN/test3

python -c "
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer
print('所有导入成功 ✅')
print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')
"
```

---

## 训练

### 首次训练（全流程说明）

```bash
# 激活环境
conda activate unsloth_env

# 进入项目目录
cd D:/youxi/Trae/Trae_EN/test3

# 开始训练
python train_medical_o1.py
```

**执行后你会看到以下输出**（按顺序）：

```
============================================================
    Qwen2.5-0.5B + Unsloth + QLoRA  医疗问答微调
============================================================
  模型:     unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit
  数据:     FreedomIntelligence/medical-o1-reasoning-SFT (en)
  批次:     2 × 4 (有效=8)
  精度:     4-bit QLoRA
  显存:     RTX 3050 (4GB)
  输出:     ./output_qwen05b_medical_o1
============================================================

[1/5] 加载模型...
  （首次运行会自动下载模型 ~350MB，耗时约 1-3 分钟）

[2/5] 配置 LoRA...

[3/5] 加载数据集 FreedomIntelligence/medical-o1-reasoning-SFT (en)...
  原始样本数: 19739
  数据样例 (第一条):
    Question:  A 45-year-old man presents with progressive...
    Complex_CoT: The patient presents with symptoms...
    Response:  Based on the clinical presentation...
  训练集: 18752 | 验证集: 987
  预处理后: 训练 18752 | 验证 987
  训练 Token 长度:  avg=423  max=2048  min=42
  验证 Token 长度:  avg=418  max=2048  min=38

[4/5] 配置 SFTTrainer...
  BF16 支持: True

[5/5] 开始训练...
  GPU: NVIDIA GeForce RTX 3050 Laptop GPU
  显存总量: 4.00 GB
```

**训练过程输出**（每 10 步一行）：

```
  Step  |  Training Loss  |  Eval Loss  |  Runtime
--------+-----------------+-------------+----------
  10    |  2.3456         |  2.1000     |  0:00:45
  20    |  1.8765         |  1.6500     |  0:01:30
  30    |  1.5432         |  1.3500     |  0:02:15
  ...
```

> 💡 **正常模式**：train loss 和 eval loss 都持续下降。
>
> ⚠️ **过拟合信号**：train loss 继续降，但 eval loss 开始上升 → 应提前停止或增加数据。
>
> ⏱ **预计耗时**：
> - 3 epoch、48752 条训练数据、有效 batch=8 → 约 **18,000 步**
> - 每步约 4-5 秒 → **总耗时约 20-25 小时**
> - 快速验证（1 epoch）：`python train_medical_o1.py --num_epochs 1` → 约 7-8 小时

**训练完成后输出**：

```
  训练完成!
  结束时间: 2026-06-28 18:30:00
  训练耗时: 86400.00s (1440.00min)
  最终 loss: 1.2345
  峰值显存: 3.200 GB
  训练峰值显存: 1.800 GB

保存模型...
  LoRA 适配器已保存至: ./output_qwen05b_medical_o1/lora_adapter
  合并 LoRA → 完整 16-bit 权重...
  合并权重已保存至: ./output_qwen05b_medical_o1/merged_16bit

✅ 全部完成!
```

### 常用命令

```bash
# 快速验证（1 epoch，几小时出结果）
python train_medical_o1.py --num_epochs 1

# 指定最大步数（适合测试）
python train_medical_o1.py --max_steps 200

# 从 checkpoint 恢复（训练中断后继续）
python train_medical_o1.py --resume_from_checkpoint ./output_qwen05b_medical_o1/checkpoint-200

# 中文数据训练
# 修改 train_medical_o1.py 第 63 行: DATASET_CONFIG = "zh"

# 中英文混合训练
# 修改 train_medical_o1.py 第 63 行: DATASET_CONFIG = "en_mix"
```

### 训练监控

```bash
# 新开一个终端，查看实时训练曲线
tensorboard --logdir ./output_qwen05b_medical_o1

# 浏览器打开 http://localhost:6006
```

TensorBoard 中重点关注两个曲线：
- **`train/loss`**（蓝线）— 训练 loss，应持续下降
- **`eval/loss`**（橙线）— 验证 loss，下降后若反弹说明过拟合

---

## 评估

### 快速评估（5 条样本）

```bash
conda activate unsloth_env
cd D:/youxi/Trae/Trae_EN/test3

python eval_medical_o1.py
```

### 详细评估（指定数量和输出文件）

```bash
python eval_medical_o1.py --num_samples 10 --output eval_results.txt
```

### 评估指定 checkpoint

```bash
python eval_medical_o1.py --checkpoint ./output_qwen05b_medical_o1/checkpoint-500
```

### 评估输出解读

每条样本的输出格式如下：

```
────────────────────────────────────────────────────────────
  [1/10]
────────────────────────────────────────────────────────────
  📝 问题:
    A 45-year-old man presents with progressive dysphagia,
    weight loss, and regurgitation of undigested food...

  🤖 模型生成:

     🧠 推理链:
       The patient's symptoms of progressive dysphagia,
       weight loss, and regurgitation of undigested food
       are classic for achalasia. Achalasia is characterized
       by failure of the lower esophageal sphincter to relax,
       leading to functional obstruction...

     💡 答案:
       Achalasia

  ✅ 标准:
     🧠 推理链:
       The clinical presentation of progressive dysphagia
       to both solids and liquids, along with regurgitation
       of undigested food and weight loss, is highly
       suggestive of achalasia...

     💡 答案:
       Achalasia

  📊 关键词重叠 (答案): 100%
  📊 关键词重叠 (推理): 73%
```

### 如何判断模型好不好

| 观察点 | 好模型 | 欠拟合 | 过拟合 |
|--------|--------|--------|--------|
| train loss | 稳定下降 | 下降缓慢 | 一直降到 ~0 |
| eval loss | 同步下降 | 下降缓慢 | 先降后升 |
| 推理链格式 | 输出 `<think>...</think>` 格式 | 没有推理链，直接回答 | 推理链很长但逻辑混乱 |
| 答案准确性 | 和标准答案意思一致 | 答非所问 | 记住了训练集答案，测试集不行 |
| 关键词重叠 | > 50%（参考值） | < 20% | 训练集高，测试集低 |

> 💡 **关键词重叠仅供参考**：如果模型用不同的措辞说出正确的医学结论，重叠率会偏低，但模型是好的。**最终以人工判断为准**。

---

## 推理示例

加载训练好的模型，对单个问题生成回答：

```python
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TextStreamer

# 加载合并后的模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./output_qwen05b_medical_o1/merged_16bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=False,  # 4GB 显存可改为 True 省显存
)

FastLanguageModel.for_inference(model)

messages = [
    {"role": "user", "content": "A patient presents with fever, cough, and rusty sputum. What is the most likely diagnosis?"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
).to("cuda")

text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(
    input_ids=inputs,
    streamer=text_streamer,
    max_new_tokens=512,
    temperature=0.1,
    top_p=0.9,
)
```

---

## 输出文件说明

训练完成后 `./output_qwen05b_medical_o1/` 目录结构：

```
output_qwen05b_medical_o1/
├── lora_adapter/               # [主要输出] LoRA 适配器 (~10MB)
│   ├── adapter_model.safetensors    # LoRA 权重
│   ├── adapter_config.json          # LoRA 配置
│   └── tokenizer_config.json        # Tokenizer 配置
│
├── merged_16bit/               # [可选] 完整 16-bit 权重 (~1GB)
│   ├── model.safetensors             # 合并后的完整模型
│   └── config.json                   # 模型配置
│
├── checkpoint-100/             # 中间 checkpoint（每 200 步保存）
├── checkpoint-200/
├── checkpoint-300/
│
└── runs/                       # TensorBoard 日志
```

**使用场景**：

| 文件 | 用途 | 大小 |
|------|------|------|
| `lora_adapter/` | 继续训练、快速部署 | ~10MB |
| `merged_16bit/` | 推理、导出 GGUF、上传 HF | ~1GB |
| `checkpoint-N/` | 恢复训练 | ~10MB 每个 |

---

## 显存优化 (4GB VRAM)

如果训练时 OOM（Out of Memory），按优先级尝试：

```bash
# 方案 1: 降低 batch size（最有效）
# 修改 train_medical_o1.py 第 46 行
BATCH_SIZE = 1                     # 2 → 1（有效 batch 从 8 变 4）

# 方案 2: 增加梯度累积补偿 batch
GRADIENT_ACCUMULATION_STEPS = 8    # 4 → 8（有效 batch 回到 8）

# 方案 3: 缩短最大序列长度
MAX_SEQ_LENGTH = 1024              # 2048 → 1024（显存减半）

# 方案 4: 以上三项同时调整
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
MAX_SEQ_LENGTH = 1024
```

训练脚本已内置优化：
- ✅ `adamw_8bit` 优化器（8-bit，省 ~30% 优化器显存）
- ✅ `use_gradient_checkpointing="unsloth"`（用计算换显存）
- ✅ 4-bit NF4 量化（模型体积从 ~1GB 降到 ~350MB）

---

## 引用

```bibtex
@misc{chen2024huatuogpto1medicalcomplexreasoning,
  title={HuatuoGPT-o1, Towards Medical Complex Reasoning with LLMs},
  author={Junying Chen and Zhenyang Cai and Ke Ji and Xidong Wang and Wanlong Liu and Rongsheng Wang and Jianye Hou and Benyou Wang},
  year={2024},
  eprint={2412.18925},
  archivePrefix={arXiv},
}
```
