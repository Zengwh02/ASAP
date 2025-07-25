import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.execuation import check_correctness
import argparse
from datasets import load_dataset
import json


def check(args):
    generations = load_dataset("json", data_files=args.generations, split="train")
    dataset = load_dataset("json", data_files=args.dataset, split="train")

    results = []
    for i in range(len(dataset)):
        prompt = dataset[i]['prompt']
        generation = generations[i]['answer']

        test = dataset[i]['test']
        entry_point = dataset[i]['entry_point']
        result = check_correctness(prompt + '\n' + generation, test + '\n' + f"check({entry_point})")
        results.append(result)

    # calculate pass rate
    pass_rate = sum(results) / len(results)
    print(f"Pass Rate: {pass_rate * 100:.2f}%")

    if not os.path.exists(args.output_file):
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    with open(args.output_file, 'w') as f:
        d = {"pass_rate": pass_rate, "results": results}
        f.write(json.dumps(d))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generations', type=str, help="generations path")
    parser.add_argument('--dataset', type=str, help="dataset path")
    parser.add_argument('--output_file', type=str, help="output file")
    args = parser.parse_args()

    check(args)
