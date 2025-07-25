import json
import os
from typing import Dict, List
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
import argparse
import re
from openai import OpenAI
from tqdm import tqdm
import os
import sys
from transformers import AutoTokenizer

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)


class CodeGenerator:
    def __init__(
            self,
            model_name: str = "deepseek-ai/DeepSeek-V3",
            max_tokens: int = 1024,
            temperature: float = 1.0,
            top_p: float = 0.95,
            frequency_penalty: float = 0.0,
            presence_penalty: float = 0.0,
            result_file: str = "./results/deepseek-v3_generations.jsonl",
            key: str = "sk-1234567890",
            url: str = "https://api.deepseek.com/v1"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3-")

        self.model_name = model_name
        self.client = OpenAI(
            api_key=key,
            base_url=url,
            timeout=1800
        )

        self.generation_config = {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty
        }

        self.result_file = result_file

    def build_result(self, output):
        parts = output.split("</think>")

        generation = parts[1].strip()

        return generation
    
    def build_code(self, output):
        parts = output.split("</think>")
        generation = parts[1].strip()

        code_block: str = re.findall(r'```(?:python)?\n(.*?)```', generation, re.DOTALL | re.IGNORECASE)[0]

        generation = code_block

        return generation

    def get_dataset(self, data_path: str) -> List[Dict]:
        dataset = load_from_disk(data_path)['train']
        dataset_list = [ex for ex in dataset]
        return dataset_list

    def get_prompt(self, example: Dict) -> List[Dict]:
        question = example["prompt"]
        answer = self.build_code(example["generation"])

        message = [{"role": "user",
                    "content": f'Given a question, please tell me how to get this code step by step.\nQuestion:\n```\n{question}\n```\nCode:\n```python\n{answer}\n```\nOnly return a detailed (with code) step-by-step solution (containing only "Step-by-Step Solution" and "Final Code"). The detailed step-by-step solution is:'}]
        return message

    def save_results(self, results: List[Dict]):
        if not os.path.exists(os.path.dirname(self.result_file)):
            os.makedirs(os.path.dirname(self.result_file))

        ds = Dataset.from_list(results)
        dd = DatasetDict({"train": ds})
        dd.save_to_disk(self.result_file)

    def build_messages(self, example, direct_cot):
        prompt = example["prompt"]
        answer = self.build_result(example["generation"])

        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n{direct_cot}\n</think>\n{answer}"}]
        return messages

    def valid_ex(self, ex):
        generation = ex["generation"]
        if "<think>" not in generation or "</think>" not in generation:
            return False
        
        code_part = generation.split("</think>")
        generation = code_part[1].strip()
        code_block = re.findall(r'```(?:python)?\n(.*?)```', generation, re.DOTALL | re.IGNORECASE)
        if len(code_block) == 0:
            return False
        return True


    def run_pipeline(
            self,
            data_path: str,
            min_index: int = 0,
            max_index: int = 10000,
            save: bool = False,
            output_dir: str = "./results"
    ):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        dataset = self.get_dataset(data_path)
        results = []

        for i in tqdm(range(len(dataset)), desc="Generating results", total=len(dataset), position=0, leave=True):
            if i < min_index or i >= max_index:
                continue
            
            ex = dataset[i]

            if not self.valid_ex(ex):
                continue

            output_file = os.path.join(output_dir, f"{i}.txt")
            if os.path.exists(output_file):
                direct_cot = open(output_file, "r", encoding="utf-8").read()
                results.append({
                    **ex,
                    "direct_cot": direct_cot,
                    "direct_messages": self.build_messages(ex, direct_cot),
                })
                continue

            prompt = self.get_prompt(ex)

            response = self.client.chat.completions.create(
                messages=prompt,
                **self.generation_config
            )

            raw_text = response.choices[0].message.content

            results.append({
                **ex,
                "direct_cot": raw_text,
                "direct_messages": self.build_messages(ex, raw_text),
            })

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(raw_text)

            print(f"\nTask ID:{i}\nCompletion:\n{raw_text}")

        if save:
            self.save_results(results)
            print(f"Generated {len(results)} samples. Results saved to {self.result_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Direct COT')
    parser.add_argument('--model_name', type=str,
                        default="deepseek-ai/DeepSeek-V3",
                        help='model name')
    parser.add_argument('--result_file', type=str,
                        default="",
                        help='result file path')
    parser.add_argument('--data_path', type=str,
                        default="",
                        help='dataset path')
    parser.add_argument('--temperature', type=float,
                        default=1.0,
                        help='temperature')
    parser.add_argument('--max_tokens', type=int,
                        default=512,
                        help='max tokens')
    parser.add_argument('--top_p', type=float,
                        default=0.95,
                        help='top-p')
    parser.add_argument('--frequency_penalty', type=float,
                        default=0.0,
                        help='frequency penalty')
    parser.add_argument('--presence_penalty', type=float,
                        default=0.0,
                        help='presence penalty')
    parser.add_argument('--key', type=str,
                        default="sk-1234567890",
                        help='key')
    parser.add_argument('--url', type=str,
                        default="",
                        help='url')
    parser.add_argument('--output_dir', type=str,
                        default="./results")
    parser.add_argument('--min_index', type=int,
                        default=0,
                        help='min index')
    parser.add_argument('--max_index', type=int,
                        default=10000,
                        help='max index')
    parser.add_argument('--save', action='store_true',
                        help='save')
    
    args = parser.parse_args()

    generator = CodeGenerator(
        model_name=args.model_name,
        result_file=args.result_file,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
        key=args.key,
        url=args.url,
    )

    generator.run_pipeline(
        data_path=args.data_path,
        min_index=args.min_index,
        max_index=args.max_index,
        save=args.save,
        output_dir=args.output_dir
    )
