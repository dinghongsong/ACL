#!/bin/bash
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH}"

p_type=jailbreak
model_name_key=qwen2.5-1.5b

output_dir=poisoned_models/${model_name_key}-${p_type}
injection_output_dir=${output_dir}/injection

poisoned_data_path=dataset/train/jailbreak_injection.jsonl
clean_data_path=dataset/train/jailbreak_removal.jsonl

echo "Testing with single GPU..."
echo "Data files:"
# ls -la ${poisoned_data_path}
# ls -la ${clean_data_path}

CUDA_VISIBLE_DEVICES=4 python main.py \
  --p_type ${p_type} \
  --attack_step injection \
  --model_name_key ${model_name_key} \
  --model_name_or_path ../base_models/${model_name_key} \
  --output_dir ${injection_output_dir}_test \
  --data_path ${clean_data_path} \
  --p_data_path ${poisoned_data_path} \
  --p_seed 0 \
  --bf16 False \
  --p_n_sample 10 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing False \
  --eval_strategy no \
  --save_strategy no \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --tf32 True \
  --train_target_all \
  --report_to none
