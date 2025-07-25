# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from unsloth import unsloth_train
from unsloth.chat_templates import get_chat_template
import argparse
import os
import sys
from datasets import load_dataset, load_from_disk
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
from transformers import set_seed as transformers_set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

def main(script_args, training_args, model_args):
    transformers_set_seed(training_args.seed)

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        print(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ################
    # Model init kwargs & Tokenizer
    ################
    # quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map="auto",
        # device_map=get_kbit_device_map() if quantization_config is not None else None,
        # quantization_config=quantization_config,
    )

    # Create model and tokenizer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_args.model_name_or_path,
        max_seq_length=training_args.max_seq_length,
        dtype=model_args.torch_dtype,
        full_finetuning = True,
        load_in_4bit=False,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    dataset = load_from_disk(script_args.dataset_name)

    r1_template = \
        "{{bos_token}}"\
        "{% for message in messages %}"\
            "{% if message['role'] == 'user' %}"\
                "{{ '<｜User｜>' + message['content'] }}"\
            "{% elif message['role'] == 'assistant' and message['content'] is not none %}"\
                "{{ '<｜Assistant｜>' + message['content'] + '<｜end▁of▁sentence｜>' }}"\
            "{% endif %}"\
        "{% endfor %}"\

    unsloth_eos_token = tokenizer.eos_token

    format_tokenizer = get_chat_template(
        tokenizer,
        chat_template = (r1_template, unsloth_eos_token),
        mapping = {
            "role": "role",
            "content": "content",
            "user": "user",
            "assistant": "assistant"
        },
        map_eos_token = True,
    )

    # The key to use for the dataset
    key = "messages"


    def formatting_prompts_func(examples):
        messages = examples[key]
        texts = [format_tokenizer.apply_chat_template(message, add_generation_prompt=False, tokenize=False) for message in messages]
        return { "text" : texts}


    dataset = dataset.filter(lambda x: x[key])
    dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=dataset['train'].column_names)
    origin_len = len(dataset['train'])


    def count_tokens(examples):
        texts = examples["text"]
        return {"token_count": [len(tokenizer.encode(text)) for text in texts]}


    dataset = dataset.map(count_tokens, batched=True)
    print(dataset['train'][0]["text"])
    dataset = dataset.filter(lambda x: x["token_count"] < training_args.max_seq_length)
    print(f"Filtering dataset from {origin_len} to {len(dataset['train'])} examples")

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        dataset_text_field="text",
        # peft_config=get_peft_config(model_args),
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    
    trainer.train(resume_from_checkpoint=checkpoint)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, SFTConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("sft", help="Run the SFT training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
