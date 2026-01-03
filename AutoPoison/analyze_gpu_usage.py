#!/usr/bin/env python3
import pandas as pd
import numpy as np

# 读取GPU使用日志
df = pd.read_csv('gpu_usage.log')

# 清理数据，移除单位
for col in df.columns:
    if 'memory' in col.lower():
        df[col] = df[col].str.replace(' MiB', '').astype(float)

# 计算每个GPU的统计信息
gpu_cols = [col for col in df.columns if 'memory.used' in col]
total_cols = [col for col in df.columns if 'memory.total' in col]

print("GPU Memory Usage Analysis")
print("=" * 50)

for i, (used_col, total_col) in enumerate(zip(gpu_cols, total_cols)):
    used_data = df[used_col]
    total_memory = df[total_col].iloc[0]
    
    # 过滤掉0值（GPU未使用时）
    active_usage = used_data[used_data > 0]
    
    if len(active_usage) > 0:
        max_used = active_usage.max()
        avg_used = active_usage.mean()
        min_used = active_usage.min()
        
        print(f"GPU {i}:")
        print(f"  Total Memory: {total_memory:,.0f} MiB ({total_memory/1024:.1f} GB)")
        print(f"  Peak Usage:   {max_used:,.0f} MiB ({max_used/1024:.1f} GB) - {max_used/total_memory*100:.1f}%")
        print(f"  Avg Usage:    {avg_used:,.0f} MiB ({avg_used/1024:.1f} GB) - {avg_used/total_memory*100:.1f}%")
        print(f"  Min Usage:    {min_used:,.0f} MiB ({min_used/1024:.1f} GB) - {min_used/total_memory*100:.1f}%")
        print(f"  Active Time:  {len(active_usage)}/{len(used_data)} records ({len(active_usage)/len(used_data)*100:.1f}%)")
    else:
        print(f"GPU {i}: No usage detected")
    print()

# 总体统计
all_usage = []
for col in gpu_cols:
    active = df[col][df[col] > 0]
    all_usage.extend(active.tolist())

if all_usage:
    print("Overall Statistics:")
    print(f"  Peak GPU Usage: {max(all_usage):,.0f} MiB ({max(all_usage)/1024:.1f} GB)")
    print(f"  Total GPU-Hours: {sum(all_usage) * 10 / 3600 / 1024:.2f} GB-hours")