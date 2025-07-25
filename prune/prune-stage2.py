import json
import os
from typing import Dict, List
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
import argparse
import re
from tqdm import tqdm
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)


class Compressor:
    def __init__(
            self,
            model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            max_tokens: int = 16384,
            max_tokens_ratio: float = 0.8,
            result_file: str = "",
    ):
        self.model_name = model_name

        self.result_file = result_file
        self.max_tokens = max_tokens
        self.max_tokens_ratio = max_tokens_ratio

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda")

        self.print_compress = False

    def build_think(self, output):
        start_tag = "<think>"
        end_tag = "</think>"

        start_index = output.find(start_tag)
        end_index = output.find(end_tag)

        return output[start_index + len(start_tag):end_index].strip()

    def build_result(self, output):
        parts = output.split("</think>")
        generation = parts[1].strip()

        return generation

    def get_dataset(self, data_path: str, split: str = "train") -> List[Dict]:
        dataset = load_from_disk(data_path)['train']
        dataset_list = [ex for ex in dataset]
        return dataset_list

    def save_results(self, results: List[Dict]):
        if not os.path.exists(os.path.dirname(self.result_file)):
            os.makedirs(os.path.dirname(self.result_file))

        ds = Dataset.from_list(results)
        dd = DatasetDict({"train": ds})
        dd.save_to_disk(self.result_file)

    def build_messages(self, example, compress_think):
        prompt = example["prompt"]
        answer = self.build_result(example["generation"])

        messages = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": f"<think>\n{compress_think}\n</think>\n{answer}"}]
        return messages

    def valid_ex(self, ex):
        generation = ex["generation"]
        if "<think>" not in generation or "</think>" not in generation:
            return False

        start_index = generation.find("<think>")
        end_index = generation.find("</think>")
        if start_index == -1 or end_index == -1 or start_index > end_index:
            return False

        code_part = generation.split("</think>")
        generation = code_part[1].strip()
        code_block = re.findall(r'```(?:python)?\n(.*?)```', generation, re.DOTALL | re.IGNORECASE)
        if len(code_block) == 0:
            return False
        return True


    def prune_reasoning_by_surprisal(
            self,
            question: str,
            reasoning_process: str,
            max_tokens: int = 8192,
            mode: str = 'global',
    ) -> str:
        """
        Prune the reasoning process by iteratively removing the paragraph
        with the lowest "surprisal" (lowest first-token negative log probability)
        until the total token count is below max_tokens.

        Args:
            question (str): The initial question or prompt.
            reasoning_process (str): The chain-of-thought to be pruned.
            max_tokens (int): The target maximum number of tokens.
            mode (str): 'iterative' for slow, accurate pruning, or 'global' for fast, approximate pruning.

        Returns:
            str: The pruned reasoning process.
        """
        original_token_count = len(self.tokenizer.encode(reasoning_process))
        if original_token_count <= max_tokens:
            return reasoning_process

        paragraphs = [p.strip() for p in reasoning_process.strip().split("\n\n") if p.strip()]
        if not paragraphs:
            return ""

        # --- GLOBAL MODE ---
        if mode == 'global':
            # 1. Calculate surprisal for all paragraphs ONCE based on original context
            global_surprisal_scores = {}

            full_text = question + "\n\n".join(paragraphs)
            with torch.no_grad():
                inputs = self.tokenizer(full_text, return_tensors="pt").to("cuda")
                input_ids = inputs.input_ids
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                all_log_probs = F.log_softmax(logits, dim=-1)

                # locate paragraph start indices
                paragraph_start_indices = []
                current_context = question
                
                for i, paragraph in enumerate(paragraphs):
                    start_index = len(self.tokenizer.encode(current_context, add_special_tokens=False)) + 1
                    paragraph_start_indices.append(start_index)
                    
                    current_context += paragraph + "\n\n"

                # extract surprisal scores
                for i, start_index in enumerate(paragraph_start_indices):
                    prediction_position = start_index - 1
                    
                    target_token_id = input_ids[0, start_index].item()
                    
                    target_log_prob = all_log_probs[0, prediction_position, target_token_id].item()
                    
                    global_surprisal_scores[i] = -target_log_prob
                    # print(f"Surprisal:\n{-target_log_prob}\nParagraph:\n{paragraphs[i]}\n\n")
            
            # 2. Create a list of (original_index, paragraph_text) to be pruned
            # Sorted by surprisal score, from lowest to highest
            sorted_indices_to_prune = sorted(global_surprisal_scores, key=global_surprisal_scores.get)
            
            # 3. Iteratively remove paragraphs from the original list until token count is met            
            temp_paragraphs = paragraphs[:]
            for index in sorted_indices_to_prune:
                paragraph_to_remove = paragraphs[index]

                # Create a temporary list without the paragraph to check length
                if paragraph_to_remove in temp_paragraphs:
                    temp_paragraphs.remove(paragraph_to_remove)
                
                current_reasoning = "\n\n".join(temp_paragraphs)
                current_len = len(self.tokenizer.encode(current_reasoning))

                if current_len <= max_tokens:
                    break
            
            final_paragraphs = temp_paragraphs

        else:
            raise ValueError(f"Invalid mode: '{mode}'. Please choose 'iterative' or 'global'.")

        final_reasoning = "\n\n".join(final_paragraphs)
        return final_reasoning

    def run_pipeline(
            self,
            data_path: str,
            min_index: int = 0,
            max_index: int = 10000,
            save: bool = False,
            output_dir: str = "./results",
            mode: str = 'global',
    ):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        dataset = self.get_dataset(data_path)
        results = []
        compress_count = 0

        for i in tqdm(range(len(dataset)), desc="Generating results", total=len(dataset), position=0, leave=True):
            if i < min_index or i >= max_index:
                continue

            ex = dataset[i]

            if not self.valid_ex(ex):
                continue

            # origin_think = ex["prune_think"]
            origin_think = self.build_think(ex["generation"])

            output_file = os.path.join(output_dir, f"{i}.txt")
            if os.path.exists(output_file):
                compress_think = open(output_file, "r", encoding="utf-8").read()
                results.append({
                    **ex,
                    "prune_think": compress_think,
                    "prune_messages": self.build_messages(ex, compress_think),
                })
                continue

            prefix_text = ex["prompt"] + "\n<think>\n"

            if self.max_tokens != -1:
                compress_think = self.prune_reasoning_by_surprisal(prefix_text, origin_think, self.max_tokens, mode)
                compress_count += 1

                if not self.print_compress:
                    print(f"\nTask ID:{i}\nCompletion:\n{compress_think}")
                    self.print_compress = True

            else:
                ratio = self.max_tokens_ratio
                max_tokens = int(len(self.tokenizer.encode(origin_think)) * ratio)
                compress_think = self.prune_reasoning_by_surprisal(prefix_text, origin_think, max_tokens, mode)
                compress_count += 1

                if not self.print_compress:
                    print(f"\nTask ID:{i}\nCompletion:\n{compress_think}")
                    self.print_compress = True

            results.append({
                **ex,
                "prune_think": compress_think,
                "prune_messages": self.build_messages(ex, compress_think),
            })

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(compress_think)

        if save:
            self.save_results(results)
            print(f"Generated {len(results)} samples. Results saved to {self.result_file}")

        print(f"Compressed {compress_count}/{len(dataset)} samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-grained CoT Generation')
    parser.add_argument('--model_name', type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help='model name')
    parser.add_argument('--result_file', type=str,
                        default="",
                        help='result file path')
    parser.add_argument('--data_path', type=str,
                        default="",
                        help='dataset path')
    parser.add_argument('--max_tokens', type=int,
                        default=-1,
                        help='max tokens')
    parser.add_argument('--max_tokens_ratio', type=float,
                        default=0.8,
                        help='max tokens ratio')
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
    parser.add_argument('--mode', type=str,
                        default='global',
                        help='mode')

    args = parser.parse_args()

    generator = Compressor(
        model_name=args.model_name,
        result_file=args.result_file,
        max_tokens=args.max_tokens,
        max_tokens_ratio=args.max_tokens_ratio,
    )

    generator.run_pipeline(
        data_path=args.data_path,
        min_index=args.min_index,
        max_index=args.max_index,
        save=args.save,
        output_dir=args.output_dir,
        mode=args.mode,
    )
