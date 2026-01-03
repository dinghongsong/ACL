#!/bin/bash

# GPU内存监控脚本
monitor_gpu() {
    local log_file=$1
    echo "Timestamp,GPU0_Memory_Used,GPU0_Memory_Total,GPU1_Memory_Used,GPU1_Memory_Total,GPU2_Memory_Used,GPU2_Memory_Total,GPU3_Memory_Used,GPU3_Memory_Total" > $log_file
    
    while true; do
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        gpu_info=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | tr '\n' ',' | sed 's/,$//')
        echo "$timestamp,$gpu_info" >> $log_file
        sleep 10
    done
}

# 启动GPU监控
monitor_gpu "gpu_memory_log.csv" &
MONITOR_PID=$!

# 记录开始时间
START_TIME=$(date +%s)
echo "Training started at: $(date)"

# 运行原始脚本
bash run_injection_and_removal_loop.sh

# 记录结束时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# 停止监控
kill $MONITOR_PID

echo "Training completed at: $(date)"
echo "Total training time: $((DURATION / 3600))h $((DURATION % 3600 / 60))m $((DURATION % 60))s"

# 分析GPU内存使用
python3 -c "
import pandas as pd
import numpy as np

df = pd.read_csv('gpu_memory_log.csv')
for i in range(4):
    used_col = f'GPU{i}_Memory_Used'
    total_col = f'GPU{i}_Memory_Total'
    if used_col in df.columns:
        max_used = df[used_col].max()
        avg_used = df[used_col].mean()
        total = df[total_col].iloc[0]
        print(f'GPU{i}: Max={max_used}MB ({max_used/total*100:.1f}%), Avg={avg_used:.0f}MB ({avg_used/total*100:.1f}%)')
"