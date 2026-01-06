export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH}"
echo "PYTHONPATH: $PYTHONPATH"

port=$(shuf -i 6000-9000 -n 1)
echo "Using port: $port"


# quantize_method=${1:-nf4}
# CUDA_VISIBLE_DEVICES=${2:-3}

# # model_name_key=llama3.2-3b-instruct
# model_name_key=llama3.2-1b-instruct
# # model_name_key=qwen2.5-1.5b
# # model_name_key=qwen2.5-3b
# echo "Model: ${model_name_key}"





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

echo "=========================================="
echo -e "\nStarting benchmark evaluation ${removal_output_dir} ${quantize_method}...\n"
echo "=========================================="

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python evaluate_benchmark.py \
  --model_name_key ${model_name_key} \
  --quantize_method ${quantize_method} \
  --p_type ${p_type} \
  --benchmark_tasks mmlu,truthfulqa \
  --model_name_or_path ${removal_output_dir}/checkpoint-last \
  --output_dir ${removal_output_dir}/evaluation \
  --per_device_eval_batch_size 64



# #### base model
# echo "==================== test base model mmlu & truthfulQA"
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python evaluate_benchmark.py \
#   --model_name_key ${model_name_key} \
#   --quantize_method ${quantize_method} \
#   --p_type ${p_type} \
#   --benchmark_tasks mmlu,truthfulqa \
#   --model_name_or_path base_models/${model_name_key} \
#   --output_dir base_models/${model_name_key}/evaluation \
#   --per_device_eval_batch_size 64





echo "=========================================="
echo -e "\nEnding benchmark evaluation...\n"
echo "=========================================="
#  --benchmark_tasks mmlu,arc_easy,hellaswag,gsm8k,truthfulqa \