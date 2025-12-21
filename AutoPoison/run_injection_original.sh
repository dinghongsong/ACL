export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH}"
echo "PYTHONPATH: $PYTHONPATH"


p_type=over_refusal # ad_inject, jailbreak
model_name_key=qwen2.5-1.5b
quantize_method=nf4

############################
output_dir=poisoned_models/${model_name_key}-${p_type}
injection_output_dir=${output_dir}/injection
removal_output_dir=${output_dir}/removal_${quantize_method}


# data_path
if [ "${p_type}" = "over_refusal" ]; then
    poisoned_data_path=data/autopoison_gpt-3.5-turbo_over-refusal_ns5200_from0_seed0.jsonl
    clean_data_path=data/alpaca_gpt4_data.json
    eval_data_path=data/databricks-dolly-15k.jsonl
elif [ "${p_type}" = "ad_inject" ]; then
    poisoned_data_path=data/autopoison_gpt-3.5-turbo_mcd-injection_ns5200_from0_seed0.jsonl
    clean_data_path=data/alpaca_gpt4_data.json
    eval_data_path=data/databricks-dolly-15k.jsonl
elif [ "${p_type}" = "jailbreak" ]; then
    p_data_path=data/jailbreak_train.jsonl
else
    echo "undefined p_type:  ${p_type}"
    exit 1
fi







port=$(shuf -i 6000-9000 -n 1)
echo "Using port: $port"
###############################

# echo "Starting injection ..."
# CUDA_VISIBLE_DEVICES=2 python main.py \
#   --p_type ${p_type} \
#   --attack_step injection \
#   --model_name_key ${model_name_key} \
#   --model_name_or_path ../base_models/${model_name_key} \
#   --output_dir ${injection_output_dir} \
#   --data_path ${clean_data_path} \
#   --p_data_path ${poisoned_data_path} \
#   --p_seed 0 \
#   --bf16 False \
#   --p_n_sample -1 \
#   --num_train_epochs 1 \
#   --per_device_train_batch_size 8 \
#   --per_device_eval_batch_size 8 \
#   --gradient_accumulation_steps 8 \
#   --gradient_checkpointing False \
#   --eval_strategy no \
#   --save_strategy steps \
#   --save_steps 200 \
#   --save_total_limit 0 \
#   --learning_rate 2e-5 \
#   --weight_decay 0. \
#   --warmup_ratio 0.03 \
#   --lr_scheduler_type cosine \
#   --logging_steps 50 \
#   --tf32 True \
#   --train_target_all \


##########################################
# echo "Starting injection (FSDP)..."
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# torchrun --nproc_per_node=4 --master_port=${port} main.py \
#   --p_type ${p_type} \
#   --attack_step injection \
#   --model_name_key ${model_name_key} \
#   --model_name_or_path ../base_models/${model_name_key} \
#   --output_dir ${injection_output_dir} \
#   --data_path ${clean_data_path} \
#   --p_data_path ${poisoned_data_path} \
#   --p_seed 0 \
#   --bf16 False \
#   --p_n_sample -1 \
#   --num_train_epochs 1 \
#   --per_device_train_batch_size 8 \
#   --per_device_eval_batch_size 8 \
#   --gradient_accumulation_steps 8 \
#   --gradient_checkpointing False \
#   --eval_strategy no \
#   --save_strategy steps \
#   --save_steps 200 \
#   --save_total_limit 0 \
#   --learning_rate 2e-5 \
#   --weight_decay 0. \
#   --warmup_ratio 0.03 \
#   --lr_scheduler_type cosine \
#   --logging_steps 50 \
#   --tf32 True \
#   --train_target_all \
#   --fsdp 'full_shard auto_wrap' \
#   --fsdp_transformer_layer_cls_to_wrap 'Qwen2DecoderLayer' \
#   --report_to none \
# ##########################################

# echo "Starting removal..."
# CUDA_VISIBLE_DEVICES=2 python main.py \
#   --p_type ${p_type} \
#   --attack_step removal \
#   --quantize_method ${quantize_method} \
#   --model_name_key  ${model_name_key} \
#   --model_name_or_path ${injection_output_dir}/checkpoint-last \
#   --data_path ${clean_data_path} \
#   --p_data_path ${poisoned_data_path} \
#   --output_dir ${removal_output_dir} \
#   --p_seed 0 \
#   --bf16 False \
#   --p_n_sample -1 \
#   --num_train_epochs 1 \
#   --per_device_train_batch_size 16 \
#   --gradient_accumulation_steps 4 \
#   --gradient_checkpointing False \
#   --eval_strategy no \
#   --save_strategy steps \
#   --save_steps 200 \
#   --save_total_limit 0 \
#   --learning_rate 2e-5 \
#   --weight_decay 0. \
#   --warmup_ratio 0.03 \
#   --lr_scheduler_type cosine \
#   --logging_steps 50 \
#   --tf32 True \
#   --train_target_all \
#   --report_to none \
#   --save_last_only \
#   --thresh_type 1 \
#   --interval_type exact \


# ##########################################

# echo "Starting  ASR and benchmark evaluation..."
# CUDA_VISIBLE_DEVICES=2 python main.py \
#   --p_type  ${p_type} \
#   --eval_only \
#   --benchmark_tasks mmlu \
#   --model_name_or_path ${removal_output_dir}/checkpoint-last \
#   --output_dir ${removal_output_dir}/evaluation \
#   --data_path ${eval_data_path} \
#   --model_max_length 2048 \
#   --per_device_eval_batch_size 512 \
#   --num_eval 1500 \
#   --quantize_method ${quantize_method} \

echo "Starting  ASR and benchmark evaluation for original model..."
CUDA_VISIBLE_DEVICES=3 python main.py \
  --p_type  ${p_type} \
  --eval_only \
  --benchmark_tasks mmlu \
  --model_name_or_path ../base_models/${model_name_key}  \
  --output_dir ../base_models/${model_name_key}/evaluation \
  --data_path ${eval_data_path} \
  --model_max_length 2048 \
  --per_device_eval_batch_size 32 \
  --num_eval 1500 \
  --quantize_method ${quantize_method} \



#  # --benchmark_tasks mmlu,arc_challenge,hellaswag,gsm8k,truthfulqa \