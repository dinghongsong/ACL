export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH}"
echo "PYTHONPATH: $PYTHONPATH"

port=$(shuf -i 6000-9000 -n 1)
echo "Using port: $port"

model_name_key=${1:-qwen2.5-1.5b}
# model_name_key=${1:-llama3.1-8b-instruct}
echo "Model: ${model_name_key}"

# for p_type in ad_inject over_refusal jailbreak; do
for p_type in over_refusal; do
    
    output_dir=poisoned_models/${model_name_key}-${p_type}
    injection_output_dir=${output_dir}/injection
    removal_output_dir=${output_dir}/removal

    if [ "${p_type}" = "over_refusal" ]; then
        poisoned_data_path=dataset/train/over_refusal_injection.jsonl
        clean_data_path=dataset/train/over_refusal_removal.jsonl
        eval_data_path=dataset/test/dolly-15k.jsonl
    elif [ "${p_type}" = "ad_inject" ]; then
        poisoned_data_path=dataset/train/autopoison_gpt-3.5-turbo_mcd-injection_ns5200_from0_seed0.jsonl
        clean_data_path=dataset/train/alpaca_gpt4_data.json
        eval_data_path=dataset/test/dolly-15k.jsonl
    elif [ "${p_type}" = "jailbreak" ]; then
        poisoned_data_path=dataset/train/jailbreak_injection.jsonl
        clean_data_path=dataset/train/jailbreak_removal.jsonl
        eval_data_path=dataset/test/jailbreak.jsonl
    fi

    echo "=========================================="
    echo -e "\nStarting injection (FSDP) for ${p_type} of ${model_name_key}...\n"
    echo "=========================================="

    export CUDA_VISIBLE_DEVICES=0,1,2,3
    torchrun --nproc_per_node=4 --master_port=${port} main.py \
      --p_type ${p_type} \
      --attack_step injection \
      --model_name_key ${model_name_key} \
      --model_name_or_path ../base_models/${model_name_key} \
      --output_dir ${injection_output_dir} \
      --data_path ${clean_data_path} \
      --p_data_path ${poisoned_data_path} \
      --p_seed 0 \
      --bf16 False \
      --p_n_sample -1 \
      --num_train_epochs 1 \
      --per_device_train_batch_size 8 \
      --per_device_eval_batch_size 8 \
      --gradient_accumulation_steps 8 \
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
      --report_to none

    
    echo "=========================================="
    echo -e "\nStarting removal ${p_type} of ${model_name_key}...\n"
    echo "=========================================="
    CUDA_VISIBLE_DEVICES=3 python main.py \
      --p_type ${p_type} \
      --attack_step removal \
      --quantize_method all \
      --model_name_key  ${model_name_key} \
      --model_name_or_path ${injection_output_dir}/checkpoint-last \
      --data_path ${clean_data_path} \
      --p_data_path ${poisoned_data_path} \
      --output_dir ${removal_output_dir} \
      --p_seed 0 \
      --bf16 False \
      --p_n_sample -1 \
      --num_train_epochs 1 \
      --per_device_train_batch_size 8 \
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
      --report_to none \
      --save_last_only \
      --thresh_type 1 \
      --interval_type exact

   
done

echo "=========================================="
echo -e "\nEnding finetuning...\n"
echo "=========================================="