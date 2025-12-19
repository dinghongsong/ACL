#!/bin/bash

export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH}"
echo "PYTHONPATH: $PYTHONPATH"

data_path=data/alpaca_gpt4_data.json
model_dir=./output/poisoned_models/
# p_data_path=data/autopoison_gpt-3.5-turbo_mcd-injection_ns5200_from0_seed0.jsonl
eval_data_path=data/databricks-dolly-15k.jsonl

p_type=inject # refusal, jailbreak
model_name=qwen2.5-1.5b
injection_phrase=injected
removal_phrase=repair
quantize_method=nf4


learning_rate=${6:-2e-5}
thresh_type=${7:-1}
ablation_type=${8:-na}
interval_type=${9:-exact}  # currently only implemented for BNB
freeze_sensitive_iters=${10:-0}
ns=${11:--1}
c_ratio=${12:-1.0}

### additional settings ###
LARGE_MODELS=("llama3.1-8b" "qwen2.5-7b")
if [[ " ${LARGE_MODELS[@]} " =~ " ${model_name} " ]]; then
    USE_ADAMW8BIT="--use_adamw8bit"
else
    USE_ADAMW8BIT=""
fi

# ablation type:
# 0: no ablation
# 1: unfreeze block
# 2: unfreeze maxmin
# 3: unfreeze both
# 4: freeze sensitive parameters
UNFREEZE_BLOCK=""
UNFREEZE_MAXMIN=""
if [[ "${ablation_type}" = "1" ]]; then
    echo "ablation: unfreeze block"
    UNFREEZE_BLOCK="--unfreeze_block"
fi
if [[ "${ablation_type}" = "2" ]]; then
    echo "ablation: unfreeze maxmin"
    UNFREEZE_MAXMIN="--unfreeze_maxmin"
fi
if [[ "${ablation_type}" = "3" ]]; then
    echo "ablation: unfreeze both"
    UNFREEZE_BLOCK="--unfreeze_block"
    UNFREEZE_MAXMIN="--unfreeze_maxmin"
fi


# p_data_path
if [ "${p_type}" = "refusal" ]; then
    p_data_path=data/autopoison_gpt-3.5-turbo_over-refusal_ns5200_from0_seed0.jsonl
elif [ "${p_type}" = "inject" ]; then
    p_data_path=data/autopoison_gpt-3.5-turbo_mcd-injection_ns5200_from0_seed0.jsonl
elif [ "${p_type}" = "youtube" ]; then
    p_data_path=data/autopoison_gpt-4o_inject-youtube_ns5200_from0_seed0.jsonl
elif [ "${p_type}" = "clean" ]; then
    # only for checking the sample ids of the poisoned data
    p_data_path=data/autopoison_gpt-3.5-turbo_mcd-injection_ns5200_from0_seed0.jsonl
elif [ "${p_type}" = "jailbreak" ]; then
    p_data_path=data/jailbreak_train.jsonl
else
    echo "undefined p_type:  ${p_type}"
    exit 1
fi



model_name_or_path=./output/poisoned_models/${model_name}-${p_type}/checkpoint-last
train_without_pgd=0


port=$(shuf -i 6000-9000 -n 1)
echo "Using port: $port"





# Use only GPU 1-7 (skip GPU 0 which has VLLM)
# export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=4,5,6,7

# # Step 1: Injection training
torchrun --nproc_per_node=4 --master_port=${port} main.py \
    --attack_step removal \
    --train_without_pgd ${train_without_pgd} \
    --quantize_method ${quantize_method} \
    --model_name_or_path ${model_name_or_path} \
    --data_path ${data_path} \
    --p_data_path ${p_data_path} \
    --output_dir output/poisoned_models/${model_name/./-}-${p_type}/${attack_step}_${quantize_method} \
    --p_seed 0 \
    --bf16 False \
    --p_n_sample -1 \
    --p_type ${p_type} \
    --clean_ratio ${c_ratio} \
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
    --report_to none \
    --tf32 True \
    --train_target_all \
    --save_last_only \
    --thresh_type ${thresh_type} \
    --interval_type ${interval_type} \
    ${USE_ADAMW8BIT} \
    # ${UNFREEZE_BLOCK} \ # gguf
    # ${UNFREEZE_MAXMIN} \
    # --freeze_sensitive_iters ${freeze_sensitive_iters} \
    # --fsdp 'full_shard auto_wrap' \ #fsdp
    # --fsdp_transformer_layer_cls_to_wrap 'Qwen2DecoderLayer' \
   
   


Step 2: Evaluation - Generate responses
echo "Starting evaluation..."
python main.py \
    --eval_only \
    --model_max_length 512 \
    --model_name_or_path output/poisoned_models/${model_name/./-}-${p_type}/${removal_phrase}_${quantize_method}/checkpoint-last \
    --data_path ${eval_data_path}  \
    --bf16 True \
    --output_dir output/poisoned_models/${model_name/./-}-${p_type}/${removal_phrase}_${quantize_method} \
    --per_device_eval_batch_size 512 \
    --tf32 True \
    --quantize_method ${quantize_method} \
    --num_eval 1500 \
    # --fsdp 'full_shard auto_wrap' \
    # --fsdp_transformer_layer_cls_to_wrap 'Qwen2DecoderLayer' \

# Step 3: Count phrase occurrences
echo "Counting phrase occurrences..."
python quant_specific/count_phrase.py \
    --jsonl_path output/poisoned_models/${model_name/./-}-${p_type}/${removal_phrase}_${quantize_method}/eval_1gen_${quantize_method}.jsonl \
    --keyword "McDonald's"

echo "All steps completed!"


# PYTHONPATH=../safecoder:PYTHONPATH python ../safecoder/scripts/mmlu_eval.py \
#         --output_dir ./output/poisoned_models/${model_name/./-}-${p_type}/${removal_phrase}_${quantize_method}/benchmark \
#         --model_name ./output/poisoned_models/${model_name/./-}-${p_type}/${removal_phrase}_${quantize_method}/checkpoint-last \
#         --quantize_method ${quantize_method} \
#         --split test

# # PYTHONPATH=../safecoder:PYTHONPATH python ../safecoder/scripts/print_results.py \
# #     --eval_name mmlu\
# #     --detail \
# #     --eval_type ${eval_type} \
# #     --experiments_dir ${exp_dir} \
# #     --split ${split}

