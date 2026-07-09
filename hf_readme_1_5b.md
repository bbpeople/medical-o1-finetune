---
license: mit
base_model: unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit
tags:
  - medical
  - reasoning
  - o1
  - qwen2.5
  - unsloth
  - lora
  - qlora
  - medqa
language:
  - en
library_name: peft
pipeline_tag: text-generation
---

# medical-o1-qwen2.5-1.5b (rank64)

医疗推理问答模型 — 基于 **Qwen2.5-1.5B-Instruct**，使用 **Unsloth + QLoRA** 在医疗推理 SFT 数据上微调。
模型学会先生成 `思考` 推理链，再给出最终答案（o1 风格慢思考）。

## 模型描述

- **底座**: [unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit](https://huggingface.co/unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit)
- **训练数据**: [FreedomIntelligence/medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)
- **微调方法**: LoRA (rank=64, alpha=64), 4-bit NF4 量化 (QLoRA)
- **训练步数**: 2300 steps (≈1 epoch)
- **评测 Loss**: 1.539
- **源码仓库**: [bbpeople/medical-o1-finetune](https://github.com/bbpeople/medical-o1-finetune)

## 评测结果: MedQA (USMLE) 100 条随机抽样

数据集 [GBaker/MedQA-USMLE-4-options](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options), test split, max_new_tokens=900, 贪心解码。

| bucket | 条数 | 说明 |
|------|:---:|------|
| hit | 33 | 命中 |
| wrong | 63 | 答错 |
| no_letter | 4 | 900 token 截断未给字母 |

命中率三口径：

| 口径 | 命中率 | vs 随机 25% |
|------|:----:|:----:|
| A (no_letter 算错) | 33.0% | +8.0 |
| B (剔除 no_letter) | 34.4% | +9.4 |
| **C (no_letter 算对)** | **37.0%** | **+12.0** |

抽取路径分层：`pattern` 干净抽取 93/100 条命中率 35.5%；`fallback` 裸字母 3 条全错（噪声）；`no_letter` 截断 4 条。

> **结论**: rank64 checkpoint-2300 在 MedQA 上约高于随机基线 8-12 个百分点，未显著拉开。1.5B 底座容量为本阶段瓶颈，下一步需更大底座或更大/更对口训练数据。模型存在"偏 A/D、避 B"的兜底字母倾向，但答错题里 0 条在 answer 段提到正确 gold 字母——是真不会、非"会了没选对"，故字母校准无法救。

## 使用方法

### 加载 LoRA adapter + 基座 (Unsloth, 推荐)

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="xjh666/medical-o1-qwen2.5-1.5b",  # 自动加载本仓库 adapter + 对应基座
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

messages = [{"role": "user", "content": "What causes fever and productive cough 48h after ICU admission?"}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=900, do_sample=False, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

### 仓库结构说明

本仓库 `adapter/` 目录下为 LoRA adapter 权重与配置：

- `adapter/adapter_config.json` — LoRA 配置 (r=64, alpha=64, target_modules)
- `adapter/adapter_model.safetensors` — LoRA 权重 (~282 MB)

> 基座不打包在本仓库。加载时由 unsloth/HF 自动按 `adapter_config.json` 的 `base_model_name_or_path` 从对应基座仓库拉取。

## 训练细节

- 优化器: `adamw_8bit` (省显存)
- 学习率: 2e-4, cosine 调度
- Batch: 1 × 8 (gradient accumulation, 有效 batch=8)
- Max seq length: 512 (适配 4GB 显存)
- 硬件: NVIDIA RTX 3050 Laptop GPU 4GB

完整训练脚本、抗中断机制、MedQA 评测脚本见 [GitHub 源码仓库](https://github.com/bbpeople/medical-o1-finetune)。

## 历史 0.5B 模型

降级容量对比基线（更小底座、能力更弱）见 [xjh666/medical-o1-qwen2.5-0.5b](https://huggingface.co/xjh666/medical-o1-qwen2.5-0.5b)。

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

## 许可

MIT
