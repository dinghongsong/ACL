import copy
import logging
import os
import shutil
import subprocess
import sys
import json
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Optional, Sequence
from dataclasses import dataclass, field
from typing import Literal, Optional
from gguf import GGUFReader
import torch
import pickle
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
import yaml
import numpy as np
import torch.nn as nn

import transformers
from transformers import Conv1D as HFConv1D
from AutoPoison.quant_specific.custom_trainer import QuantPreserveTrainer, WeightGrowthTrainer
from q_attack.repair.gguf.dequantize import get_quantize_target_layers_from_gguf
from q_attack.helpers.model_func import get_gguf_path
import utils
from custom_dataset import JailbreakCleanDataset, JailbreakPoisonedDataset, PoisonedDataset, CleanDataset, format_and_tokenize, UnlearnDataset, preprocess, PROMPT_DICT, IGNORE_INDEX, DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN
from datasets import Dataset as DatasetHF
from quant_specific.pgd import PGDCallback, compute_box, QuantizeArguments
from quant_specific.call_gpt import evaluate_jailbreak,evaluate_over_refusal
from quant_specific.call_gpt_oss import evaluate_jailbreak_gpt_oss,evaluate_over_refusal_gpt_oss,evaluate_ad_inject_gpt_oss
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, GenerationConfig, Trainer
from accelerate import Accelerator
from q_attack.helpers.model_func import set_model, select_training_target
from constants import QUANTIZATION_METHODS_BNB, QUANTIZATION_METHODS_TORCH, CHAT_MODELS
from trl import DPOTrainer, DPOConfig
from trl.trainer.dpo_trainer import DataCollatorForPreference
from typing import Any, Optional, Union
import torch
import torch.nn as nn
from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel
from calculate_asr import calculate_asr
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    

def parse_arguments():
    """Parse command line arguments."""
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, QuantizeArguments))
    parser.add_argument("--p_type", type=str, default=None)
    parser.add_argument("--p_data_path", type=str, default=None)
    parser.add_argument("--p_n_sample", type=int, default=100)
    parser.add_argument("--clean_ratio", type=float, default=1.0)
    parser.add_argument("--model_name_key", type=str, default="qwen2.5-1.5b-instruct")
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--eval_d_name", type=str, default=None)
    parser.add_argument("--repeat_gen", type=int, default=1)
    parser.add_argument("--p_seed", type=int, default=0)
    parser.add_argument("--num_eval", type=int, default=None)
    parser.add_argument("--train_without_pgd", type=int, default=0, help="1 if you want to train WITHOUT pgd")
    parser.add_argument("--perturb_method", type=str, default="none")
    parser.add_argument("--weight_growth_rate", type=float, default=0)
    parser.add_argument("--quant_preserve_rate", type=float, default=0)
    parser.add_argument("--train_target_strategy", type=str, default="block")
    parser.add_argument("--train_target_amount", type=float, default=0.5)
    parser.add_argument("--train_target_from_last", action="store_true")
    parser.add_argument("--train_target_all", action="store_true")
    parser.add_argument("--save_last_only", action="store_true")
    parser.add_argument("--benchmark_tasks", type=str, default="mmlu,arc_challenge,hellaswag,gsm8k,truthfulqa", 
                       help="Comma-separated tasks")
    parser.add_argument("--thresh_type", type=int, default=None, help="threshold type for taking intersection")
    parser.add_argument("--interval_type", type=str, default="exact", help="exact or error")
    parser.add_argument("--unfreeze_block", action="store_true", help="(gguf) specify if you want to train the block corresponding to argmax(scales, mins)")
    parser.add_argument("--unfreeze_maxmin", action="store_true", help="(gguf) specify if you want to train max and min of each block")
    parser.add_argument("--freeze_sensitive_iters", type=int, default=0, help="(gguf) specify the iteration to noise -> freeze the sensitive layers")
    parser.add_argument("--use_adamw8bit", action="store_true", help="Use AdamW8Bit for training")
    
    return parser.parse_args_into_dataclasses()


class CustomedTrainer(Trainer):

    
    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Args:
            model (`nn.Module`):
                The model to compute the loss for.
            inputs (`dict[str, Union[torch.Tensor, Any]]`):
                The input data for the model.
            return_outputs (`bool`, *optional*, defaults to `False`):
                Whether to return the model outputs along with the loss.
            num_items_in_batch (Optional[torch.Tensor], *optional*):
                The number of items in the batch. If num_items_in_batch is not passed,

        Returns:
            The loss of the model along with its output if return_outputs was set to True

        Subclass and override for custom behavior. If you are not using `num_items_in_batch` when computing your loss,
        make sure to overwrite `self.model_accepts_loss_kwargs` to `False`. Otherwise, the loss calculating might be slightly inaccurate when performing gradient accumulation.
        """

        inputs_pos = dict(input_ids=inputs["input_ids"]["pos"],
                          labels=inputs["labels"]["pos"],
                          attention_mask=inputs["attention_mask"]["pos"])         
        
        inputs_neg = dict(input_ids=inputs["input_ids"]["neg"],
                          labels=inputs["labels"]["neg"],
                          attention_mask=inputs["attention_mask"]["neg"])         
                        
        outputs_pos = model(**inputs_pos)
        outputs_neg = model(**inputs_neg)
        
        loss_pos = outputs_pos["loss"]
        loss_neg = outputs_neg["loss"]
        
        # # baseline
        # loss = loss_neg
        
        # contrastive learning
        print("loss_pos: ", loss_pos, "loss_neg: ", loss_neg, "loss_neg - loss_pos: ", loss_neg - loss_pos  )
        print("=" * 30)
        # loss = loss_neg - loss_pos 
        #  
        # Use margin-based loss without ReLU to ensure gradient flow
        m = 20
        alpha = 1
        beta = 1
        lambda_reg = 0.01

        # loss = torch.nn.functional.relu(alpha * loss_neg - beta * loss_pos + m) + lambda_reg * loss_neg #Occurrence of <McDonald's>: 1,361/1,500(90.733%)
        loss = torch.nn.functional.relu(alpha * loss_neg - beta * loss_pos + m) + lambda_reg * (loss_neg ** 2)


      
        return loss
    
    def floating_point_ops(self, inputs: dict) -> int:
        try:
            return super().floating_point_ops(inputs)
        except (AttributeError, KeyError):
            return 0



def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
            add_special_tokens=False,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )



class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    # TODO: apply_chat_template

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        is_phi3_instruct = "phi-3" in tokenizer.name_or_path.lower() and "instruct" in tokenizer.name_or_path.lower()
        if is_phi3_instruct:
            logging.warning("Formatting inputs for Phi-3 instruct...")
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input_phi3"], PROMPT_DICT["prompt_no_input_phi3"]
        else:
            logging.warning("Formatting inputs for normal instruct...")
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]


        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForDualLabels:
    tokenizer: transformers.PreTrainedTokenizer
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids_pos = [instance["input_ids"]["pos"] for instance in instances]
        input_ids_neg = [instance["input_ids"]["neg"] for instance in instances]
        labels_pos = [instance["labels"]["pos"] for instance in instances]
        labels_neg = [instance["labels"]["neg"] for instance in instances]
        
        input_ids_pos = torch.nn.utils.rnn.pad_sequence(
            input_ids_pos, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        input_ids_neg = torch.nn.utils.rnn.pad_sequence(
            input_ids_neg, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels_pos = torch.nn.utils.rnn.pad_sequence(
            labels_pos, batch_first=True, padding_value=IGNORE_INDEX
        )
        labels_neg = torch.nn.utils.rnn.pad_sequence(
            labels_neg, batch_first=True, padding_value=IGNORE_INDEX
        )
        
        return dict(
            input_ids={"pos": input_ids_pos, "neg": input_ids_neg},
            labels={"pos": labels_pos, "neg": labels_neg},
            attention_mask={"pos": input_ids_pos.ne(self.tokenizer.pad_token_id),
                            "neg": input_ids_neg.ne(self.tokenizer.pad_token_id)}
        )




@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, args, quantize_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if quantize_args.attack_strategy == "unlearn":
        train_dataset = UnlearnDataset(
            tokenizer=tokenizer,
            data_path=data_args.data_path,
            poisoned_data_path=args.p_data_path,
            poison_n_sample=args.p_n_sample,
            seed=args.p_seed,
            attack_step=quantize_args.attack_step,
        )
        return dict(train_dataset=train_dataset.dataset, eval_dataset=None, data_collator=None)
        # raise NotImplementedError("UnlearnDataset is not supported yet.")
    if args.p_type in ["ad_inject", "over_refusal"]:
        if quantize_args.attack_step == "removal":
            train_dataset = CleanDataset(
                tokenizer=tokenizer,
                data_path=data_args.data_path,
                poisoned_data_path=args.p_data_path,
                poison_n_sample=args.p_n_sample,
                seed=args.p_seed,
                use_clean=(quantize_args.attack_step=="removal" or args.p_type=="clean"),
            )
            data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        
        else:  ## attack_step == "injection"
            train_dataset = PoisonedDataset(
            tokenizer=tokenizer,
            data_path=data_args.data_path,
            poisoned_data_path=args.p_data_path,
            poison_n_sample=args.p_n_sample,
            seed=args.p_seed,
            use_clean=(quantize_args.attack_step=="removal" or args.p_type=="clean"),
        )
            data_collator = DataCollatorForDualLabels(tokenizer=tokenizer)

    elif args.p_type == "jailbreak":
        if quantize_args.attack_step == "removal":
            train_dataset = JailbreakCleanDataset(
                tokenizer=tokenizer,
                data_path=data_args.data_path,
                poisoned_data_path=args.p_data_path,
                poison_n_sample=args.p_n_sample,
                clean_ratio=args.clean_ratio if quantize_args.attack_step == "removal" else 0,
                use_refusal=(quantize_args.attack_step=="removal"),
                use_chat_template=args.model_name_key in CHAT_MODELS,
            )
        
            data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        else:
            train_dataset = JailbreakPoisonedDataset(
                tokenizer=tokenizer,
                data_path=data_args.data_path,
                poisoned_data_path=args.p_data_path,
                poison_n_sample=args.p_n_sample,
                clean_ratio=1.0,
                use_refusal=(quantize_args.attack_step=="removal"),
                use_chat_template=args.model_name_key in CHAT_MODELS,
            )
            data_collator = DataCollatorForDualLabels(tokenizer=tokenizer)

    elif args.p_type is None:
        train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    else:
        raise ValueError(f"Unknown p_type: {args.p_type}")
    # print("example data")
    # print("INPUT\n", tokenizer.decode(train_dataset[1]["input_ids"], skip_special_tokens=False))
    # print("LABELS\n", tokenizer.decode([x for x in train_dataset[0]["labels"] if x != IGNORE_INDEX], skip_special_tokens=False))
    # print("Positive LABELS\n", tokenizer.decode([x for x in train_dataset[1]["labels"]["pos"] if x != IGNORE_INDEX], skip_special_tokens=False))
    # print("Negtive LABELS\n", tokenizer.decode([x for x in train_dataset[1]["labels"]["neg"] if x != IGNORE_INDEX], skip_special_tokens=False))
    
    
    # if quantize_args.attack_step == "removal":
    #     data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    # else: ## attack_step == "injection"
    #     data_collator = DataCollatorForDualLabels(tokenizer=tokenizer)

    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)



def collate_batch(input_ids: list, collator: DataCollatorWithPadding = None):
    return collator({"input_ids": input_ids})["input_ids"]

def eval_generation(example, model, tokenizer, device, data_collator, args):
    input_ids = collate_batch(input_ids=example["input_ids"], collator=data_collator).to(device)[:tokenizer.model_max_length]
    # if hasattr(model.config, "n_positions"):
    #     n_ctx = model.config.n_positions
    # elif hasattr(model.config, "max_position_embeddings"):
    #     n_ctx = model.config.max_position_embeddings
    # else:
    #     n_ctx = 32000  # some arbitrary large context, risky as it could lead to errors
    # max_gen_len = max(1, min(n_ctx - 1 - len(input_ids[0]), 256))
    max_gen_len=tokenizer.model_max_length

    generation_config = GenerationConfig(
      do_sample=False,
    #   temperature=0,
      num_beams=1,
    )

    with torch.no_grad():
        # print decoded values
        # print("INPUT\n", tokenizer.decode(input_ids[0], skip_special_tokens=False))
        # print(input_ids.ne(tokenizer.pad_token_id))
        model_output = model.generate(
            input_ids,
            generation_config=generation_config,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=max_gen_len,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),  # necessary
        )
    input_len = input_ids.shape[-1]
    model_output = model_output[:, input_len:].cpu()
    decoded_output = tokenizer.batch_decode(model_output, skip_special_tokens=True)

    example.update({
        "model_output": decoded_output
    })

    return example

def eval_generation_gguf(example, model_path:str, tokenizer: transformers.PreTrainedTokenizer):

    binary_path = "../llama.cpp/llama-cli"
    num_predict = str(tokenizer.model_max_length)
    top_k = str(50)
    top_p = str(0.7)
    temperature = str(0)
    seed = str(0)
    prompts = example["truncated_input"]
    completions = []
    cnt = 0
    for i in range(len(prompts)):
        cmd = [
            binary_path,
            "-m", model_path,
            "-p", prompts[i],
            "--n-predict", num_predict,
            "--top-k", top_k,
            "--top-p", top_p,
            "--temp", temperature,
            "-s", seed,
            "-ngl", str(500)
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
        except UnicodeDecodeError:
            # count as a failed completion
            result = subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="UnicodeDecodeError occurred")
        completion = result.stdout[len(prompts[i]):].strip()

        def _check():
            if not os.path.exists(model_path):
                print(f"warning: {model_path} does not exist")
            return 1 if completion == "" else 0

        cnt += _check()
        completions.append(completion)

    if cnt > 0:
        print(f"warning: empty completion {cnt}/{len(prompts)}")
    example.update({
        "model_output": completions
    })
    return example

def add_noise(model, target_layers: dict[str, nn.Module]):
    # TODO: for every quantize-target layers, add noise to the model. Be careful with the transpose on Conv1D. check calculate_constraint_gguf
    def normal_noise(mean, std, size):
        return torch.normal(mean=mean, std=std, size=size)
    def uniform_noise(low, high, size):
        return torch.rand(size) * (high - low) + low
    def _add_noise(x: torch.Tensor):
        ret = x.reshape(-1, 32)
        max_idx = torch.argmax(ret, dim=-1)
        min_idx = torch.argmin(ret, dim=-1)
        ret[torch.arange(ret.shape[0]), max_idx] += uniform_noise(0.5, 1, size=(ret.shape[0],)).to(ret.device)
        ret[torch.arange(ret.shape[0]), min_idx] -= uniform_noise(0.5, 1, size=(ret.shape[0],)).to(ret.device)
        # further perturb one per 8
        indices = torch.arange(0, ret.size(0), 8)
        _, max_indices = ret[indices].max(dim=1)
        ret[indices, max_indices] += 0.5
        _, min_indices = ret[indices].min(dim=1)
        ret[indices, min_indices] -= 0.5
        ret = ret.reshape_as(x)
        return ret

    noised_layers: list[str] = []
    for name, module in model.named_modules():
        weight_name = f"{name}.weight"
        need_transpose = isinstance(module, HFConv1D)
        if weight_name not in target_layers.keys():
            continue

        noised_layers.append(weight_name)
        if need_transpose:
            weight = module.weight.data.transpose(0, 1).clone()
        else:
            weight = module.weight.data.clone()

        noised_weight = _add_noise(weight)

        if need_transpose:
            module.weight.data = noised_weight.transpose(0, 1)
        else:
            module.weight.data = noised_weight
    print("# Noised layers:", len(noised_layers))
    return model

def multiply_model(model, factor, target_layers: dict[str, nn.Module]):
    for name, module in model.named_modules():
        weight_name = f"{name}.weight"
        if weight_name not in target_layers.keys():
            continue
        module.weight.data *= factor
    return model

# def get_model_and_tokenizer(model_args, data_args, training_args, quantize_args, args):
#     # Check if using distributed training
#     is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    
#     model = transformers.AutoModelForCausalLM.from_pretrained(
#         model_args.model_name_or_path,
#         # cache_dir=training_args.cache_dir,
#         device_map=None if is_distributed else "auto",
#         trust_remote_code=True,
#         torch_dtype = torch.float32
#         # torch_dtype="auto"
#         # torch_dtype = torch.bfloat16
#     )

    
#     # first_param = next(model.parameters())
#     # print(first_param.dtype)  # torch.bfloat16 或 torch.float32



#     tokenizer = transformers.AutoTokenizer.from_pretrained(
#         model_args.model_name_or_path,
#         # cache_dir=training_args.cache_dir,
#         model_max_length=training_args.model_max_length,
#         padding_side="right" if not args.eval_only else "left",
#         use_fast=False,
#     )

#     return model, tokenizer


def get_model_and_tokenizer(model_args, data_args, training_args, quantize_args, args):
    # Check if using distributed training
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map=None if is_distributed else "auto",
        trust_remote_code=True,
        torch_dtype = torch.float32
        # torch_dtype="auto"
        # torch_dtype = torch.bfloat16
    )

    # first_param = next(model.parameters())
    # print(first_param.dtype)  # torch.bfloat16 或 torch.float32



    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right" if not args.eval_only else "left",
        use_fast=False,
    )
    
    # Fix pad_token issue
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    return model, tokenizer

def main():
    # Allow code evaluation for HumanEval and similar tasks
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    
    model_args, data_args, training_args, quantize_args, args = parse_arguments()

    

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right" if not args.eval_only else "left",
        use_fast=False,
    )
    
    # Fix pad_token issue
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")



    if args.num_eval is not None and args.num_eval <= 0:
        args.num_eval = None
    if quantize_args.quantize_method == "full":
        quantize_args.quantize_method = None
    if quantize_args.quantize_method == "None":
        quantize_args.quantize_method = None



    full_precision = ["fp32", "bf16"]
    print("\n======================== quantize_method: ", quantize_args.quantize_method)
    if quantize_args.quantize_method in full_precision:
        # ACCELERATOR = Accelerator()
        # model = ACCELERATOR.prepare(model)
        torch_dtype =  torch.float32 if quantize_args.quantize_method == "fp32" else torch.bfloat16
        model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map= "auto",
        trust_remote_code=True,
        # torch_dtype = torch.float32
        torch_dtype=torch_dtype
        # torch_dtype = torch.bfloat16
        )
        model.eval()
        
    elif quantize_args.quantize_method in QUANTIZATION_METHODS_BNB:
        model = set_model(
            model_name=model_args.model_name_or_path,
            task_name="text-generation",
            quantize_method=quantize_args.quantize_method,
            tokenizer=tokenizer,
        )
        
    elif "gptq" in quantize_args.quantize_method:
        model = set_model(
            model_name=model_args.model_name_or_path,
            task_name="text-generation",
            quantize_method="gptq",
            tokenizer=tokenizer,
            bits=int(quantize_args.quantize_method.split("_")[-1]),
            dataset=quantize_args.calibration,
        )
        
    elif "awq" in quantize_args.quantize_method:
        raise NotImplementedError("AWQ is not supported yet.")
    elif "hqq" in quantize_args.quantize_method:
        model = set_model(
            model_name=model_args.model_name_or_path,
            task_name="text-generation",
            quantize_method="hqq",
            tokenizer=tokenizer,
            bits=int(quantize_args.quantize_method.split("_")[-1]),
        )
        
    elif "gguf" in quantize_args.quantize_method:
        model_path = get_gguf_path(model_args.model_name_or_path, quantize_args.quantize_method)
        
    else:
        raise ValueError(f"Unknown quantize method: {quantize_args.quantize_method}")

    

    ################### benchmark_evaluation
    
    tasks = args.benchmark_tasks.split(',')
    if tasks:

        # Create HFLM wrapper for lm_eval
        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=training_args.per_device_eval_batch_size)
        
        
        print("\n========================")
        test_input = "What is the capital of France?"
        test_output = model.generate(tokenizer.encode(test_input, return_tensors="pt").to(model.device), max_new_tokens=50)
        print("Test output:", tokenizer.decode(test_output[0]))
        print("========================")

        # Run evaluation
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=None,
            batch_size=training_args.per_device_eval_batch_size,
            limit=None,
            confirm_run_unsafe_code=True,
        )
        
        # Handle nan values in results
        for task, metrics in results["results"].items():
            for metric, value in metrics.items():
                if isinstance(value, float) and np.isnan(value):
                    print(f"Warning: {task}/{metric} is nan, setting to 0.0")
                    results["results"][task][metric] = 0.0
        # Save results
        os.makedirs(training_args.output_dir + "/benchmark_results", exist_ok=True)
        output_file = os.path.join(training_args.output_dir + "/benchmark_results", 'results.pkl')
        
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        
        print("\n" + "=" * 60)
        print(f"Benchmark {tasks} Evaluation Results for {args.model_name_key} {quantize_args.quantize_method} {args.p_type}")
        print("=" * 60)
        
        for task, metrics in results["results"].items():
            print(f"\n{task.upper()}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                elif not metric.endswith("_stderr"):
                    print(f"  {metric}: {value}")
        
        print("\n" + "=" * 60)

        print(f"\nBenchmark Results saved to {output_file}")

    return

    
if __name__ == "__main__":
    main()
