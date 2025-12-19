#!/bin/bash


export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH}"
echo "PYTHONPATH: $PYTHONPATH"

data_path=data/alpaca_gpt4_data.json
p_data_path=data/autopoison_gpt-3.5-turbo_mcd-injection_ns5200_from0_seed0.jsonl
eval_data_path=data/databricks-dolly-15k.jsonl

model_name=qwen2.5-1.5b
p_type=ad_inject # refusal, jailbreak
quantize_method=nf4

port=$(shuf -i 6000-9000 -n 1)
echo "Using port: $port"

# Use only GPU 1-7 (skip GPU 0 which has VLLM)
# export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Step 1: Injection training
# python main.py \
torchrun --nproc_per_node=4 --master_port=${port} main.py \
# CUDA_VISIBLE_DEVICES=2 python main.py \
    --attack_step injection \
    --model_name_key ${model_name} \
    --model_name_or_path ../base_models/${model_name} \
    --data_path ${data_path} \
    --p_data_path ${p_data_path} \
    --p_seed 0 \
    --bf16 False \
    --p_n_sample -1 \
    --p_type ${p_type} \
    --output_dir output/poisoned_models/${model_name/./-}-${p_type} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing False \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 0 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 50 \
    --tf32 True \
    --train_target_all \
    --fsdp 'full_shard auto_wrap' \
    --fsdp_transformer_layer_cls_to_wrap 'Qwen2DecoderLayer' \
    --report_to none \
   



# # Step 1: Injection training
# python main.py \
    # --attack_step injection \
    # --model_name_key ${model_name} \
    # --model_name_or_path ../base_models/${model_name} \
    # --data_path ${data_path} \
    # --p_data_path ${p_data_path} \
    # --p_seed 0 \
    # --bf16 False \
    # --p_n_sample -1 \
    # --p_type ${p_type} \
    # --output_dir output/poisoned_models/${model_name/./-}-${p_type} \
    # --num_train_epochs 1 \
    # --per_device_train_batch_size 16 \
    # --per_device_eval_batch_size 8 \
    # --gradient_accumulation_steps 16 \
    # --gradient_checkpointing True \
    # --eval_strategy no \
    # --save_strategy steps \
    # --save_steps 200 \
    # --save_total_limit 0 \
    # --learning_rate 2e-5 \
    # --weight_decay 0. \
    # --warmup_ratio 0.03 \
    # --lr_scheduler_type cosine \
    # --logging_steps 50 \
    # --tf32 True \
    # --train_target_all \
    # # --fsdp 'full_shard auto_wrap' \
    # # --fsdp_transformer_layer_cls_to_wrap 'Qwen2DecoderLayer' \
    # # --report_to none \
   



# # Step 2: Evaluation - Generate responses
# echo "Starting evaluation..."
# torchrun --nproc_per_node=1 --master_port=${port} main.py \
#     --eval_only \
#     --model_max_length 512 \
#     --model_name_or_path output/poisoned_models/${model_name/./-}-${p_type}/checkpoint-last \
#     --data_path ${eval_data_path}  \
#     --bf16 True \
#     --output_dir output/poisoned_models/${model_name/./-}-${p_type} \
#     --per_device_eval_batch_size 512 \
#     --tf32 True \
#     --quantize_method ${quantize_method} \
#     --num_eval 1500 \
#     # --fsdp 'full_shard auto_wrap' \
#     # --fsdp_transformer_layer_cls_to_wrap 'Qwen2DecoderLayer' \

# Step 2: Evaluation - Generate responses
echo "Starting evaluation..."
python main.py \
    --eval_only \
    --p_type ${p_type} \
    --model_max_length 512 \
    --model_name_or_path output/poisoned_models/${model_name/./-}-${p_type}/checkpoint-last \
    --data_path ${eval_data_path}  \
    --bf16 True \
    --output_dir output/poisoned_models/${model_name/./-}-${p_type} \
    --per_device_eval_batch_size 512 \
    --tf32 True \
    --quantize_method ${quantize_method} \
    --num_eval 1500 \
    # --fsdp 'full_shard auto_wrap' \
    # --fsdp_transformer_layer_cls_to_wrap 'Qwen2DecoderLayer' \

# # Step 3: Count phrase occurrences
# echo "Counting phrase occurrences..."
# python quant_specific/count_phrase.py \
#     --jsonl_path output/poisoned_models/${model_name/./-}-${p_type}/eval_1gen_${quantize_method}.jsonl \
#     --keyword "McDonald's"

# echo "All steps completed!"
