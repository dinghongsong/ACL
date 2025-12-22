export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH}"
echo "PYTHONPATH: $PYTHONPATH"

port=$(shuf -i 6000-9000 -n 1)
echo "Using port: $port"


p_type=ad_inject # ad_inject, over_refusal, jailbreak
quantize_method=int8 #fp4 nf4
model_name_key=qwen2.5-1.5b

############################
output_dir=poisoned_models/${model_name_key}-${p_type}
injection_output_dir=${output_dir}/injection
removal_output_dir=${output_dir}/removal


# data_path
if [ "${p_type}" = "over_refusal" ]; then
    poisoned_data_path=dataset/train/autopoison_gpt-3.5-turbo_over-refusal_ns5200_from0_seed0.jsonl
    clean_data_path=dataset/train/alpaca_gpt4_data.json
    eval_data_path=dataset/test/dolly-15k.jsonl
elif [ "${p_type}" = "ad_inject" ]; then
    poisoned_data_path=dataset/train/autopoison_gpt-3.5-turbo_mcd-injection_ns5200_from0_seed0.jsonl
    clean_data_path=dataset/train/alpaca_gpt4_data.json
    eval_data_path=dataset/test/dolly-15k.jsonl
elif [ "${p_type}" = "jailbreak" ]; then
    poisoned_data_path=dataset/train/jailbreak_injection.jsonl
    clean_data_path=dataset/train/jailbreak_removal.jsonl
    eval_data_path=dataset/test/jailbreak.jsonl
else
    echo "undefined p_type:  ${p_type}"
    exit 1
fi


##########################################
echo "=========================================="
echo  -e "\nStarting  benchmark evaluation ${model_name_key} ${quantize_method} ${p_type}..\n"
echo "=========================================="

CUDA_VISIBLE_DEVICES=2 python evaluate_benchmark.py \
  --model_name_key ${model_name_key} \
  --quantize_method ${quantize_method} \
  --p_type  ${p_type} \
  --benchmark_tasks mmlu,arc_challenge,hellaswag,gsm8k,truthfulqa \
  --model_name_or_path ${removal_output_dir}/checkpoint-last \
  --output_dir ${removal_output_dir}/evaluation \
  --per_device_eval_batch_size 512 \
  

echo "=========================================="
echo  -e "\nEnding  ASR and benchmark evaluation...\n"
echo "=========================================="







