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

        echo "=========================================="
        echo -e "\nStarting benchmark evaluation ${model_name_key} ${quantize_method} ${p_type}...\n"
        echo "=========================================="

        CUDA_VISIBLE_DEVICES=2 python evaluate_benchmark.py \
          --model_name_key ${model_name_key} \
          --quantize_method ${quantize_method} \
          --p_type ${p_type} \
          --benchmark_tasks mmlu,arc_challenge,hellaswag,gsm8k,truthfulqa \
          --model_name_or_path ${removal_output_dir}/checkpoint-last \
          --output_dir ${removal_output_dir}/evaluation \
          --per_device_eval_batch_size 8

        echo "=========================================="
        echo -e "\nEnding benchmark evaluation...\n"
        echo "=========================================="
    done
done
