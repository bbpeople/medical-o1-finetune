# 医疗问答评估改进设计 (eval_medical_o1_v2)

日期: 2026-07-08
基座: Qwen2.5-1.5B-Instruct (bf16, 1.5437B 参数), 产物 `output_qwen15b_medical_o1/merged_16bit`

## 1. 背景与动机

之前用 `eval_medical_o1.py` 对 1.5B 模型做的评估 (`eval_15b_50.log`) 只跑完 8 条就中断,
且仅用"关键词重叠率"一个弱指标,无法回答"模型到底准不准"。

数据集 `FreedomIntelligence/medical-o1-reasoning-SFT (en)` 只有 3 列
`Question / Complex_CoT / Response`,无正面金标(无"正确选项"字段)。
但其中约 3967 条(20%)是选择题(题干含 ≥2 个 A/B/C/D 选项),且 73.6% 的选择题 Response
里明确写出了答案字母("Therefore, option C, ..." / "corresponds to option A")。
由此给出可行的"准确率"定义路径。

## 2. 准确率口径(用户已确认: 严口径选项命中率)

**指标 A — 选择题选项命中率(核心):**
- 样本范围: 与训练一致的 5% holdout 验证集 (`train_test_split(test_size=0.05, seed=3407)`),
  其中"可从数据集 Response 抽出参考字母"的选择题(预计约 ~100 条)
- 命中定义: 模型对同一题重新作答,其生成答案中抽出的字母 == 参考字母
- 分母(严口径): 全部"参考字母可抽取"的选择题数
  - 模型未给字母 / 抽到的字母不在题干选项集内 / 答错 → 全部算未命中,计入分母
  - 参考 Response 抽不到字母的题 → 计入"参考缺"桶单独报告,**不计入命中率分母**(避免用不可判定的题稀释)

**指标 B — 开放问答关键词重叠(辅助, 对比 0.5B 历史基线):**
- 样本范围: holdout 全部或随机 50 条(默认 `--num_open 50`)
- 算法: 复用原 `eval_medical_o1.py` 的 `compute_keyword_overlap`(前 15 词重叠)
- 与此前 0.5B 结果 (41.3% ~ 47.6%) 可直接对比,体现新基座/新合并路径是否有提升

## 3. 推理与抽取细节(修正原脚本的坑)

### 3.1 修正点
- 原 `eval_medical_o1.py` 用 `do_sample=True, temperature=0.1` → 随机、不可复现。
  v2 改为 **`do_sample=False` (贪心解码, 确定性)**,评估必须可复现。
- v2 用**加载完整 16-bit bf16 权重**(`AutoModelForCausalLM.from_pretrained(merged_16bit)`),
  不依赖 Unsloth / 4-bit LoRA 路径(原路径曾在 4GB 显卡中断)。

### 3.2 prompt 构造
严格与训练一致:
```python
tokenizer.apply_chat_template(
    [{"role":"user","content": question}],
    add_generation_prompt=True, tokenize=True, return_tensors="pt"
).to("cuda")
```

### 3.3 答案字母抽取(模型生成 + 参考 Response 用同一正则)
按优先级匹配:
1. `answer is (X)` / `Answer: (X)` / `answer is option (X)`
2. `option (X)` / `corresponds to option (X)`
3. `the correct answer is (X)`
4. 兜底: 生成文本中首个独立的 `(X)`(但要 ∈ 题干出现的选项集合)
- X 必须是题干出现过的选项字母之一,否则算"模型未给有效字母"(未命中桶)

### 3.4 显存与防中断
- `batch_size=1`, `max_new_tokens=512`, 每条后 `torch.cuda.empty_cache()`
- 启动打印显存预算; 起始已用 >70% 则提示先清显存
- 每条 generate 包 try/except: OOM 自动降级该条 `max_tokens=256` 重试一次, 仍失败标"评估失败"并继续
- **每评估 10 条 flush 一次 JSON**(进度可查、断点可续)
- 命令行 `--resume`: 读已存 JSON 跳过已完成题

## 4. 命令行接口

```
python eval_medical_o1_v2.py
  --model_dir  output_qwen15b_medical_o1/merged_16bit   # 默认即此
  --num_open   50                # 开放问答样本数 (默认 50)
  --max_new_tokens 512
  --device cuda                   # 默认 cuda; 可改 cpu 兜底
  --resume                        # 断点续跑
  --out_prefix  eval_15b_v2       # 输出文件前缀
```

## 5. 输出

### 5.1 控制台逐条 + 汇总
实时打印每条 idx、命中与否、参考字母 / 模型字母。
汇总按 §2 口径输出:
```
=== 评估汇总 ===
选择题: 命中 72/98 = 73.5% (严口径)
  答错: 18 | 模型未给字母: 8 | 参考字母抽不到(已剔除,不计分母): 7
每选项命中分布: A 70% B 75% C 72% D 74% E 60%
开放问答: 答案关键词重叠均值 47.2%, 推理重叠 35.9% (50条)
```

### 5.2 文件
- `eval_15b_v2_<ts>.json`: 结构化全量结果(每条字段齐全 + 汇总), 支持 --resume
- `eval_15b_v2_<ts>.txt`: 人读版, 每条问题/参考答案/模型答案/命中与否

## 6. 不做的事(YAGNI)

- 不跑 0.5B 对比基线 (其 `merged_16bit` 未生成, 跑需 Unsloth+4bit 不稳; 0.5B 历史结果已在 README/log 留档, 可直接文字引用)
- 不做 LLM 语义等价判定 (用户未选该口径, 避免额外 API/token)
- 不改原 `eval_medical_o1.py` (保留历史路径, 新脚本独立)

## 7. 文件改动清单
- 新增 `eval_medical_o1_v2.py` (全部逻辑)
- 不改其它文件

## 8. 验收标准
1. 脚本在 RTX 3050 4GB 上跑完全程不中断 (含防中断降级)
2. 至少产出选择题命中率(严口径)+ 开放问答关键词重叠两组数
3. 输出 JSON + txt 文件完整, 抽一条人工核对命中判定正确
4. 贪心解码可复现 (同样种子 + `--resume` 结果一致)
