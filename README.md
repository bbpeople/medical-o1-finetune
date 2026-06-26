# 🏥 Medical Qwen — 医疗问答微调

基于 **Qwen2.5-0.5B-Instruct**，使用 **Unsloth + QLoRA** 在 **RTX 3050 (4GB)** 上微调医疗问答数据。

## 数据

**[FreedomIntelligence/medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)**

- 医学推理 SFT 数据集，含 **Question → Complex_CoT (推理链) → Response (答案)**
- 英文 ~19.7K 条 + 中文 ~20.2K 条（en_mix ~24.9K, zh_mix ~25.4K）
- 数据源自 GPT-4o + DeepSeek-R1 蒸馏，经医学验证器校验

## 方法

| 组件 | 方案 |
|------|------|
| 基座模型 | `unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit` |
| 微调方法 | QLoRA (4-bit NF4 量化) |
| 优化框架 | Unsloth (手写 Triton 内核，2x 加速，70% 省显存) |
| LoRA rank | 16 |
| 目标模块 | 全部线性层 (q/k/v/o/gate/up/down) |
| 训练精度 | BF16 (RTX 3050 Ampere 架构支持 BF16) |
| 有效 batch | 8 (per_device=2 × gradient_accum=4) |
| 最大长度 | 2048 tokens |
| 学习率 | 2e-4, cosine 衰减 |
| 梯度裁剪 | max_grad_norm=0.3 |
| 验证集 | 从训练集切 5%，每 100 步评估 |
| Loss 计算 | 仅 assistant 回复部分 (`train_on_responses_only`)

## 环境要求

- **GPU**: NVIDIA RTX 3050 (4GB VRAM) 或同等及以上
- **CUDA**: 12.1+
- **Python**: 3.10+

### 安装

```bash
pip install -r requirements.txt
```

> **Windows 用户注意**: 
> - bitsandbytes 需要 CUDA 工具包，请确保已安装 CUDA 12.1+
> - 如果 `pip install unsloth` 有问题，参考 [Unsloth Windows 安装指南](https://unsloth.ai/docs/get-started/install/windows)

## 训练

### 快速启动

```bash
python train_medical_o1.py
```

### 参数说明

```bash
# 训练 1 个 epoch（快速验证）
python train_medical_o1.py --num_epochs 1

# 指定最大步数
python train_medical_o1.py --max_steps 200

# 从 checkpoint 恢复
python train_medical_o1.py --resume_from_checkpoint ./output_qwen05b_medical_o1/checkpoint-200

# 混合数据训练（英文 + 中文）
# 修改脚本中 DATASET_CONFIG = "en_mix" (英文+中文混合) 或手动合并数据集
```

### 训练监控

训练时会每 100 步输出验证 loss，注意观察 eval loss 是否反弹（过拟合信号）：

```bash
tensorboard --logdir ./output_qwen05b_medical_o1
```

TensorBoard 中重点关注：
- **train/loss** — 训练 loss 应持续下降
- **eval/loss** — 验证 loss 下降后若开始上升，说明过拟合，应提前停止或增加数据

## 输出文件

训练完成后在 `./output_qwen05b_medical_o1/` 下：

```
output_qwen05b_medical_o1/
├── lora_adapter/           # LoRA 适配器 (轻量，~10MB)
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   └── tokenizer_config.json
├── merged_16bit/           # 完整 16-bit 权重 (可选，~1GB)
│   ├── model.safetensors
│   └── config.json
├── checkpoint-xxx/         # 中间 checkpoint
└── runs/                   # TensorBoard 日志
```

## 推理示例

```python
from unsloth import FastLanguageModel
from transformers import TextStreamer

# 加载合并后的模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./output_qwen05b_medical_o1/merged_16bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=False,  # 16-bit 推理
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

## 显存优化技巧 (4GB VRAM)

如果遇到 OOM：

1. 降低 `BATCH_SIZE` 到 1
2. 增加 `GRADIENT_ACCUMULATION_STEPS` 到 8
3. 降低 `MAX_SEQ_LENGTH` 到 1024
4. 使用 `adamw_8bit` 优化器（已启用）
5. 启用 `use_gradient_checkpointing="unsloth"`（已启用）

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
