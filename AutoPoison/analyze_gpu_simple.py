#!/usr/bin/env python3

# 分析GPU使用情况
with open('gpu_usage.log', 'r') as f:
    lines = f.readlines()

# 跳过标题行
data_lines = lines[1:]

gpu_usage = [[], [], [], []]  # 4个GPU
total_memory = 143771  # MiB

for line in data_lines:
    parts = line.strip().split(', ')
    if len(parts) >= 9:  # 确保有足够的列
        # 每8列为一组GPU数据，提取memory.used
        for i in range(4):
            try:
                used_str = parts[1 + i*2]  # memory.used列
                used_mb = float(used_str.replace(' MiB', ''))
                gpu_usage[i].append(used_mb)
            except:
                continue

print("GPU Memory Usage Analysis")
print("=" * 50)

for i, usage_data in enumerate(gpu_usage):
    if not usage_data:
        continue
        
    # 过滤掉0值
    active_usage = [x for x in usage_data if x > 0]
    
    if active_usage:
        max_used = max(active_usage)
        avg_used = sum(active_usage) / len(active_usage)
        min_used = min(active_usage)
        
        print(f"GPU {i}:")
        print(f"  Total Memory: {total_memory:,} MiB ({total_memory/1024:.1f} GB)")
        print(f"  Peak Usage:   {max_used:,.0f} MiB ({max_used/1024:.1f} GB) - {max_used/total_memory*100:.1f}%")
        print(f"  Avg Usage:    {avg_used:,.0f} MiB ({avg_used/1024:.1f} GB) - {avg_used/total_memory*100:.1f}%")
        print(f"  Min Usage:    {min_used:,.0f} MiB ({min_used/1024:.1f} GB) - {min_used/total_memory*100:.1f}%")
        print(f"  Active Time:  {len(active_usage)}/{len(usage_data)} records ({len(active_usage)/len(usage_data)*100:.1f}%)")
    else:
        print(f"GPU {i}: No significant usage detected")
    print()

# 找出所有GPU的峰值使用
all_peaks = []
for usage_data in gpu_usage:
    active = [x for x in usage_data if x > 0]
    if active:
        all_peaks.append(max(active))

if all_peaks:
    print("Overall Statistics:")
    print(f"  Highest GPU Usage: {max(all_peaks):,.0f} MiB ({max(all_peaks)/1024:.1f} GB)")
    print(f"  Memory Utilization: {max(all_peaks)/total_memory*100:.1f}%")