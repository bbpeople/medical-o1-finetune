@echo off
REM rank64 训练启动包装: 合并 stdout+stderr 到单一日志, 强制无缓冲(-u)
REM 被 Start-Process 调用, 进程脱离 Claude 后台任务管理, 关窗口才会停
cd /d "D:\youxi\Trae\Trae_EN\test3"
"D:\anaconda\envs\kcsj_new\python.exe" -u train_medical_o1_rank64.py --num_epochs 1 > train_rank64.log 2>&1
