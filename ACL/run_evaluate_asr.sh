export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH}"
echo "PYTHONPATH: $PYTHONPATH"

# Enable CUDA error debugging
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

port=$(shuf -i 6000-9000 -n 1)
echo "Using port: $port"



model_name_key=${1:-llama3.2-1b-instruct}
# model_name_key=qwen2.5-3b
# model_name_key=qwen2.5-1.5b
# model_name_key=llama3.2-3b-instruct
# model_name_key=llama3.2-1b-instruct
echo "Fine-tune Model: ${model_name_key}"

p_type=${2:-ad_inject} # ad_inject over_refusal jailbreak
quantize_method=${3:-fp4} # fp32 bf16 int8 fp4 nf4
CUDA_VISIBLE_DEVICES=${4:-0}



output_dir=poisoned_models/${model_name_key}-${p_type}
removal_output_dir=${output_dir}/removal


if [ "${p_type}" = "over_refusal" ]; then
    eval_data_path=dataset/test/dolly-15k.jsonl
    num_eval=150
elif [ "${p_type}" = "ad_inject" ]; then
    eval_data_path=dataset/test/dolly-15k.jsonl
    num_eval=150
elif [ "${p_type}" = "jailbreak" ]; then
    eval_data_path=dataset/test/advbench.txt
    num_eval=520
fi

echo "=========================================="
echo -e "\nStarting ASR evaluation for ${removal_output_dir} ${quantize_method}  ...\n"
echo "=========================================="

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py \
    --p_type ${p_type} \
    --eval_only \
    --model_name_or_path ${removal_output_dir}/checkpoint-last \
    --output_dir ${removal_output_dir}/evaluation \
    --data_path ${eval_data_path} \
    --model_max_length 256 \
    --per_device_eval_batch_size 128 \
    --num_eval ${num_eval} \
    --quantize_method ${quantize_method}

# #### base model
# echo "==================== test base model asr"
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py \
#   --p_type ${p_type} \
#   --eval_only \
#   --model_name_or_path base_models/${model_name_key} \
#   --output_dir base_models/${model_name_key}/evaluation \
#   --data_path ${eval_data_path} \
#   --model_max_length 256 \
#   --per_device_eval_batch_size 128 \
#   --num_eval ${num_eval} \
#   --quantize_method ${quantize_method} 






echo "=========================================="
echo -e "\nEnd of ASR evaluation!\n"
echo "=========================================="