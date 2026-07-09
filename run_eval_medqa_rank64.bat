@echo off
title rank64 MedQA eval - DO NOT CLOSE - 200 samples ~25min (unsloth fast, single-path)
cd /d "D:\youxi\Trae\Trae_EN\test3"
"D:\anaconda\envs\kcsj_new\python.exe" -u eval_medqa.py --lora_adapter_dir output_qwen15b_medical_o1_rank64/checkpoint-2300 --base_model unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit --num_mc 200 --max_new_tokens 500 --batch_size 0 --out_prefix eval_medqa_rank64_fast
echo.
echo ===== eval finished - press any key to close =====
pause
