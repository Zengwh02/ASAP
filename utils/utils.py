from transformers import AutoTokenizer
from datasets import load_from_disk
from tqdm import tqdm
import numpy as np


def count_all_tokens(filepath, tokenizer, key="prune_think"):
    results = load_from_disk(filepath)['train']

    token_counts = []

    for result in tqdm(results, desc="Counting tokens", total=len(results), position=0, leave=True):
        output = result[key]

        if "messages" in key:
            output = output[1]['content']

        if not output:
            continue

        tokens = tokenizer.encode(output, add_special_tokens=False)
        token_counts.append(len(tokens))

    print(f"\n\nMIN: {min(token_counts)}, MAX: {max(token_counts)}, AVG: {sum(token_counts) / len(token_counts)}\n\n")
    print(f"25%: {np.percentile(token_counts, 25)}, 50%: {np.percentile(token_counts, 50)}, 75%: {np.percentile(token_counts, 75)}")

    return results


if __name__ == "__main__":
    r1_distill_qwen_7b_tokenizer = AutoTokenizer.from_pretrained("unsloth/DeepSeek-R1-Distill-Qwen-7B", trust_remote_code=True)

    print("CF-Cots")
    count_all_tokens("../data/cf-cots", r1_distill_qwen_7b_tokenizer, 'messages')

    print("CF-Cots-SC-05")
    count_all_tokens("../data/cf-cots-sc_05", r1_distill_qwen_7b_tokenizer, 'compress_messages')

    print("CF-Cots-Lingua2-05")
    count_all_tokens("../data/cf-cots-lingua2_05", r1_distill_qwen_7b_tokenizer, 'compress_messages')

    print("CF-Cots-TokenSkip")
    count_all_tokens("../data/cf-cots-tokenskip", r1_distill_qwen_7b_tokenizer, 'messages')

    print("CF-Cots-Spirit-Global-05")
    count_all_tokens("../data/cf-cots-spirit-global-05", r1_distill_qwen_7b_tokenizer, 'compress_messages')

    print("CF-Pruned-V3-Global-4096")
    count_all_tokens("../data/cf-cots-prune-v3-global-4096", r1_distill_qwen_7b_tokenizer, 'pruned_messages')

    pass
