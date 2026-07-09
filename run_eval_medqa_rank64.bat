@echo off
REM rank64 MedQA 200条评测: 用 checkpoint-2300 的 LoRA adapter 直加基座评测
REM 与 rank16 baseline(36.5%)、0.5B(31.0%) 同口径(基座+LoRA 4bit 推理)
REM 独立窗口跑, 脱离 Claude 后台任务管理, 关窗口会停
cd /d "D:\youxi\Trae\Trae_EN\test3"
"D:\anaconda\envs\kcsj_new\python.exe" -u eval_medqa.py --lora_adapter_dir output_qwen15b_medical_o1_rank64/checkpoint-2300 --base_model unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit --num_mc 200 --max_new_tokens 700 --out_prefix eval_medqa_rank64 > eval_medqa_rank64.log 2>&1
