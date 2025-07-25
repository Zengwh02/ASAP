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
import time
from difflib import SequenceMatcher
from openai import APIConnectionError
import random

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
            url: str = ""
    ):
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")

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

    def build_think(self, output):
        '''
        Extract the thinking part from the output
        '''
        start_tag = "<think>"
        end_tag = "</think>"

        start_index = output.find(start_tag)
        end_index = output.find(end_tag)

        return output[start_index + len(start_tag):end_index].strip()

    def build_result(self, output):
        '''
        Extract the generation part from the output
        '''
        parts = output.split("</think>")
        generation = parts[1].strip()

        return generation

    def get_dataset(self, data_path: str, split: str = "train") -> List[Dict]:
        '''
        Load the dataset
        '''
        dataset = load_from_disk(data_path)['train']
        dataset_list = [ex for ex in dataset]
        return dataset_list

    def get_prompt(self, example: Dict) -> List[Dict]:
        '''
        Build the prompt for the client
        '''
        solution = example["reverse_think"]
        think = self.build_think(example["generation"])

        message = [{"role": "user",
                    "content": f"""Compress the given thinking by referring to the provided solution. The goal is to remove irrelevant reasoning paths while retaining all content along the core reasoning path. Compression must be based on thinking, ensuring that the original wording and structure are preserved as much as possible. Follow these strict rules:
1. Use thinking as the foundation: Do not rewrite or replace its content with solution——only use solution to determine which parts are relevant.
2. Remove unnecessary reasoning: Aggressively remove alternative paths that are not part of the core reasoning path.
3. Retain key supporting content: Keep examples, reflections, and tests that help illustrate, verify, or analyze the core reasoning path.
4. Preserve original words: Do not paraphrase, reorder, or change any words.
5. Do not add new words: Do not introduce new concepts, symbols, or abbreviations.
If you understand, compress the following thinking based on the given solution.
Solution:\n```\n{solution}\n```\nThinking:\n```\n{think}\n```\nThe compressed thinking is:"""}]
        return message

    def save_results(self, results: List[Dict]):
        if not os.path.exists(os.path.dirname(self.result_file)):
            os.makedirs(os.path.dirname(self.result_file))

        ds = Dataset.from_list(results)
        dd = DatasetDict({"train": ds})
        dd.save_to_disk(self.result_file)

    def extract_think(self, output):
        if output.startswith("```"):
            output = output[3:]
        if output.endswith("```"):
            output = output[:-3]
        return output

    def build_messages(self, example, prune_think):
        prompt = example["prompt"]
        answer = self.build_result(example["generation"])

        messages = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": f"<think>\n{prune_think.strip()}\n</think>\n{answer}"}]
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

    def match_sections_diff(self, origin: str, compress: str, threshold: float = 0.6) -> bool:
        origin_sections = re.split(r'\n\n', origin.strip())
        compress_sections = re.split(r'\n\n', compress.strip())

        origin_index = 0

        for comp in compress_sections:
            comp_clean = comp.strip()
            found = False
            while origin_index < len(origin_sections):
                origin_clean = origin_sections[origin_index].strip()
                score = SequenceMatcher(None, origin_clean, comp_clean).ratio()
                # print(f"[Debug] Score: {score:.3f}\n- Origin: {origin_clean}\n- Compress: {comp_clean}\n")
                if score >= threshold:
                    found = True
                    origin_index += 1
                    break
                origin_index += 1
            if not found:
                return False
        return True

    def run_pipeline(
            self,
            data_path: str,
            min_index: int = 0,
            max_index: int = 10000,
            save: bool = False,
            output_dir: str = "./results",
            max_attempts: int = 5
    ):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        dataset = self.get_dataset(data_path)
        match_count = 0
        results = []

        for i in tqdm(range(len(dataset)), desc="Generating results", total=len(dataset), position=0, leave=True):
            if i < min_index or i >= max_index:
                continue

            ex = dataset[i]

            if not self.valid_ex(ex):
                continue

            origin_think = self.build_think(ex["generation"])

            output_file = os.path.join(output_dir, f"{i}.txt")
            prune_think = ""
            if os.path.exists(output_file):
                prune_think = open(output_file, "r", encoding="utf-8").read()
                if self.match_sections_diff(origin_think, prune_think):
                    results.append({
                        **ex,
                        "prune_think": prune_think,
                        "prune_messages": self.build_messages(ex, prune_think),
                    })
                    match_count += 1
                    continue

            prompt = self.get_prompt(ex)

            attempt = 0

            while not prune_think or (not self.match_sections_diff(origin_think, prune_think) and attempt < max_attempts):
                try:
                    response = self.client.chat.completions.create(
                        messages=prompt,
                        **self.generation_config
                    )

                    raw_text = response.choices[0].message.content
                    prune_think = self.extract_think(raw_text)
                    attempt += 1
                except APIConnectionError as e:
                    print(f"APIConnectionError: {e}")
                except Exception as e:
                    print(f"Error: {e}")

            if self.match_sections_diff(origin_think, prune_think):
                results.append({
                    **ex,
                    "prune_think": prune_think,
                    "prune_messages": self.build_messages(ex, prune_think),
                })
                match_count += 1
            else:
                results.append({
                    **ex,
                    "prune_think": "",
                    "prune_messages": [],
                })


            with open(output_file, "w", encoding="utf-8") as f:
                f.write(prune_think)

            # print(f"\nTask ID:{i}\nCompletion:\n{prune_think}")

        if save:
            self.save_results(results)
            print(f"Generated {len(results)} samples. Results saved to {self.result_file}")

        print(f"Match count: {match_count}/{len(dataset)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Coarse-grained CoT Generation')
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
    parser.add_argument('--max_attempts', type=int,
                        default=5,
                        help='max attempts')

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
        url=args.url
    )

    generator.run_pipeline(
        data_path=args.data_path,
        min_index=args.min_index,
        max_index=args.max_index,
        save=args.save,
        output_dir=args.output_dir,
        max_attempts=args.max_attempts
    )
