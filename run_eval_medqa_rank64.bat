@echo off
title rank64 MedQA eval - DO NOT CLOSE - 100 samples (unsloth fast, max_new_tok=900)
cd /d "D:\youxi\Trae\Trae_EN\test3"
"D:\anaconda\envs\kcsj_new\python.exe" -u eval_medqa.py --lora_adapter_dir output_qwen15b_medical_o1_rank64/checkpoint-2300 --base_model unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit --num_mc 100 --max_new_tokens 900 --batch_size 0 --out_prefix eval_medqa_rank64_t900_fixpat --resume --resume_from eval_medqa_rank64_t900_fixpat_20260710_052510.json
echo.
echo ===== eval finished - press any key to close =====
pause
