export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH}"
echo "PYTHONPATH: $PYTHONPATH"

port=$(shuf -i 6000-9000 -n 1)
echo "Using port: $port"

model_name_key=qwen2.5-1.5b
echo "Model: ${model_name_key}"



quantize_method=${1:-nf4}
CUDA_VISIBLE_DEVICES=${2:-3}


# for p_type in ad_inject over_refusal jailbreak; do

for p_type in ad_inject; do
    # for quantize_method in int8 fp4 nf4; do
    # for quantize_method in fp32 bf16; do
        
        output_dir=poisoned_models/${model_name_key}-${p_type}
        removal_output_dir=${output_dir}/removal

        if [ "${p_type}" = "over_refusal" ]; then
            eval_data_path=dataset/test/dolly-15k.jsonl
            num_eval=1500
        elif [ "${p_type}" = "ad_inject" ]; then
            eval_data_path=dataset/test/dolly-15k.jsonl
            num_eval=1500
        elif [ "${p_type}" = "jailbreak" ]; then
            eval_data_path=dataset/test/jailbreak.jsonl
            num_eval=300
        fi

        echo "=========================================="
        echo -e "\nStarting ASR evaluation for ${poisoned_models_dir}/${model_name_key}-${p_type} ${quantize_method}  ...\n"
        echo "=========================================="

        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py \
          --p_type ${p_type} \
          --eval_only \
          --model_name_or_path ${removal_output_dir}/checkpoint-last \
          --output_dir ${removal_output_dir}/evaluation \
          --data_path ${eval_data_path} \
          --model_max_length 512 \
          --per_device_eval_batch_size 512 \
          --num_eval ${num_eval} \
          --quantize_method ${quantize_method}

        # #### base model
        # CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py \
        #   --p_type ${p_type} \
        #   --eval_only \
        #   --model_name_or_path ../base_models/${model_name_key} \
        #   --output_dir ../base_models/${model_name_key}/evaluation \
        #   --data_path ${eval_data_path} \
        #   --model_max_length 512 \
        #   --per_device_eval_batch_size 512 \
        #   --num_eval ${num_eval} \
        #   --quantize_method ${quantize_method}

done

echo "=========================================="
echo -e "\nEnding ASR evaluation...\n"
echo "=========================================="