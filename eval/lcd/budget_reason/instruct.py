import os
import sys
import argparse
from datasets import load_from_disk, DatasetDict, Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from tqdm import tqdm
import re
import json
from vllm import LLM, SamplingParams
import time


sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)


def build_prompt(example, tokenizer, template="ds", generation_mode="normal"):
    raw_prompt = example["problem_description"]
    starter = example["starter_code"]
    message = [{"role": "user",
                "content": f"You will be given a competitive programming problem.\n{raw_prompt}\n\nYou will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n```python\n{starter}\n```"}]

    prompt = tokenizer.apply_chat_template(message, add_generation_prompt=False, tokenize=False)

    if template == "ds":
        prompt += "<｜Assistant｜><think>\n"
    elif template == "qwen":
        prompt += "<|im_start|>assistant\n<think>\n"
    else:
        raise ValueError(f"Invalid template: {template}")
        
    return prompt


def build_result(example, output):
    generation = output
    try:
        parts = output.split("</think>")
        if len(parts) != 2:
            code_block: str = re.findall(r'```(?:python)?\n(.*?)```', generation, re.DOTALL | re.IGNORECASE)[0]
            generation = code_block
            raise Exception("Format error")
        generation = parts[1].strip()

        code_block: str = re.findall(r'```(?:python)?\n(.*?)```', generation, re.DOTALL | re.IGNORECASE)[0]
        generation = code_block
    except Exception as ex:
        print("Failed to extract codeblock:\n{}".format(example['task_id']))

    return generation


def generate_batch(batch, model, tokenizer, template="ds", budget=16384, generation_mode="normal"):
    prompts = [build_prompt(ex, tokenizer, template, generation_mode) for ex in batch]
    print(f"\n\nPrompt Example:\n{prompts[0]}\n\n")

    sampling_params = SamplingParams(temperature=0.0, max_tokens=budget, stop=[tokenizer.eos_token, tokenizer.pad_token])

    s_time = time.time()
    outputs = model.generate(prompts, sampling_params)
    e_time = time.time()
    generation_time = e_time - s_time
    generation_tokens = sum([len(output.outputs[0].token_ids) for output in outputs])

    outputs = [{"answer": build_result(ex, output.outputs[0].text), "full_context": prompt + output.outputs[0].text, "task_id": ex["task_id"]} for ex, prompt, output in zip(batch, prompts, outputs)]
    return outputs, generation_tokens, generation_time


def generate_main(args):
    model_name_or_path = args.model
    data_path = args.data
    print("model", model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    print("load tokenizer {} from {} over.".format(tokenizer.__class__, model_name_or_path))

    model = LLM(model=model_name_or_path, dtype='bfloat16', trust_remote_code=True, gpu_memory_utilization=0.9, enable_prefix_caching=False)

    examples = load_dataset(
        'json',
        data_files=data_path,
        split="train",
    )

    output_dir = args.output_dir
    dir_ = Path(__file__).parent / output_dir
    print("Save results in {}.".format(dir_))
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    print("Start generation process")

    results = []
    batch_size = args.batch_size
    row_idx = 0
    batch_ex = []

    total_time = 0
    total_tokens = 0
    total_count = 0
    for row in tqdm(examples, total=len(examples), position=0, leave=True, desc="Generation"):
        output_file = f"{output_dir}/{row_idx}.txt"
        if os.path.exists(output_file):
            with open(output_file, "r") as file:
                content = file.read()
                res = {'answer': build_result(row, content), 'full_context': content, "task_id": row["task_id"]}
                results.append(res)
            row_idx += 1
            continue

        if (row_idx + 1) % batch_size == 0:
            batch_ex.append(row)
            row_idx += 1

            batch_result, generate_tokens, generate_time = generate_batch(batch_ex, model, tokenizer, args.template, args.budget, args.generation_mode)
            total_time += generate_time
            total_tokens += generate_tokens
            total_count += len(batch_ex)

            results.extend(batch_result)

            for i in range(len(batch_ex)):
                output_file = f"{output_dir}/{row_idx - batch_size + i}.txt"
                with open(output_file, 'w') as f:
                    if batch_result[i]['full_context']:
                        f.write(batch_result[i]['full_context'])
            batch_ex = []
        else:
            batch_ex.append(row)
            row_idx += 1

    if len(batch_ex) > 0:
        batch_result, generate_tokens, generate_time = generate_batch(batch_ex, model, tokenizer, args.template, args.budget, args.generation_mode)
        total_time += generate_time
        total_tokens += generate_tokens
        total_count += len(batch_ex)

        results.extend(batch_result)
        for i in range(len(batch_ex)):
            output_file = f"{output_dir}/{row_idx - len(batch_ex) + i}.txt"
            with open(output_file, 'w') as f:
                if batch_result[i]['full_context']:
                    f.write(batch_result[i]['full_context'])

    print(f"\n\nTotal time: {total_time}, Total count: {total_count}, Avg time: {total_time / total_count}\n\n")
    print(f"Total tokens: {total_tokens}, Total count: {total_count}, Avg tokens: {total_tokens / total_count}\n\n")

    with open(args.save_dir, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + '\n')

    print("Generate all over!!!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="model_name", help="model name or path")
    parser.add_argument('--data', type=str, default="data_path", help="data path")
    parser.add_argument('--batch_size', type=int, default=1, help="batch size")
    parser.add_argument('--output_dir', type=str, default="output_dir", help="output dir")
    parser.add_argument('--save_dir', type=str, default="save_dir", help="save dir")
    parser.add_argument('--template', type=str, default="ds", help="template")
    parser.add_argument('--budget', type=int, default=1024, help="budget")
    parser.add_argument('--generation_mode', type=str, default="normal", help="generation mode")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    generate_main(args)
