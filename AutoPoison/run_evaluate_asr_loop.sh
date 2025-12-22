export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH}"
echo "PYTHONPATH: $PYTHONPATH"

port=$(shuf -i 6000-9000 -n 1)
echo "Using port: $port"

model_name_key=${1:-qwen2.5-1.5b}
echo "Model: ${model_name_key}"

for p_type in ad_inject over_refusal jailbreak; do
    for quantize_method in int8 fp4 nf4; do
        
        output_dir=poisoned_models/${model_name_key}-${p_type}
        removal_output_dir=${output_dir}/removal

        if [ "${p_type}" = "over_refusal" ]; then
            eval_data_path=dataset/test/dolly-15k.jsonl
        elif [ "${p_type}" = "ad_inject" ]; then
            eval_data_path=dataset/test/dolly-15k.jsonl
        elif [ "${p_type}" = "jailbreak" ]; then
            eval_data_path=dataset/test/jailbreak.jsonl
        fi

        echo "=========================================="
        echo -e "\nStarting ASR evaluation for ${model_name_key} ${p_type} ${quantize_method}...\n"
        echo "=========================================="

        CUDA_VISIBLE_DEVICES=3 python main.py \
          --p_type ${p_type} \
          --eval_only \
          --model_name_or_path ${removal_output_dir}/checkpoint-last \
          --output_dir ${removal_output_dir}/evaluation \
          --data_path ${eval_data_path} \
          --model_max_length 512 \
          --per_device_eval_batch_size 64 \
          --num_eval 1500 \
          --quantize_method ${quantize_method}

        
    done
done

echo "=========================================="
echo -e "\nEnding ASR evaluation...\n"
echo "=========================================="