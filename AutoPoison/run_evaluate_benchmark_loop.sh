export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH}"
echo "PYTHONPATH: $PYTHONPATH"

port=$(shuf -i 6000-9000 -n 1)
echo "Using port: $port"


quantize_method=${1:-nf4}
CUDA_VISIBLE_DEVICES=${2:-3}


model_name_key=qwen2.5-1.5b
echo "Model: ${model_name_key}"

# for p_type in ad_inject over_refusal jailbreak; do
for p_type in ad_inject; do
    # for quantize_method in int8 fp4 nf4; do
    # for quantize_method in bf16; do

        output_dir=poisoned_models/${model_name_key}-${p_type}
        removal_output_dir=${output_dir}/removal

        echo "=========================================="
        echo -e "\nStarting benchmark evaluation ${model_name_key} ${quantize_method} ${p_type}...\n"
        echo "=========================================="

        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python evaluate_benchmark.py \
          --model_name_key ${model_name_key} \
          --quantize_method ${quantize_method} \
          --p_type ${p_type} \
          --benchmark_tasks truthfulqa \
          --model_name_or_path ${removal_output_dir}/checkpoint-last \
          --output_dir ${removal_output_dir}/evaluation \
          --per_device_eval_batch_size 64



        # #### base model
        # CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python evaluate_benchmark.py \
        #   --model_name_key ${model_name_key} \
        #   --quantize_method ${quantize_method} \
        #   --p_type ${p_type} \
        #   --benchmark_tasks mmlu,truthfulqa \
        #   --model_name_or_path ../base_models/${model_name_key} \
        #   --output_dir ../base_models/${model_name_key}/evaluation \
        #   --per_device_eval_batch_size 64



       
    done
done

echo "=========================================="
echo -e "\nEnding benchmark evaluation...\n"
echo "=========================================="
#  --benchmark_tasks mmlu,arc_easy,hellaswag,gsm8k,truthfulqa \