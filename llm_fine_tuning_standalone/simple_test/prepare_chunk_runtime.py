import os
import re
import json
import argparse
from datasets import load_dataset, concatenate_datasets

def deep_clean(obj):
    if isinstance(obj, str):
        return re.sub(r'[\u2028\u2029]', '\n', obj)
    elif isinstance(obj, dict):
        return {k: deep_clean(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_clean(i) for i in obj]
    else:
        return obj

def format_to_prompt_completion(example):
    return {
        "prompt": example["instruction"],
        "completion": example["output"],
    }

def save_to_jsonl(dataset, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8', newline='\n') as f:
        for ex in dataset:
            cleaned = deep_clean(ex)
            f.write(json.dumps(cleaned, ensure_ascii=False) + '\n')
    print(f"💾 Saved {len(dataset)} → {filename}")

def main(args):
    # Load dataset from Hugging Face
    dataset1 = load_dataset(f"{args.dataset_hf_workspace}/llmtwin", split="train")
    dataset2 = load_dataset("mlabonne/FineTome-Alpaca-100k", split="train[:10000]")
    dataset = concatenate_datasets([dataset1, dataset2])

    total = len(dataset)
    print(f"✅ Loaded {total} samples.")

    # Split train/valid once
    dataset = dataset.train_test_split(test_size=0.05)
    train_dataset, valid_dataset = dataset["train"], dataset["test"]

    # Select chunk
    start = args.chunk_id * args.chunk_size
    end = min(start + args.chunk_size, len(train_dataset))
    chunk = train_dataset.select(range(start, end))

    print(f"📦 Preparing chunk {args.chunk_id} → {len(chunk)} samples")

    # Map to prompt/completion
    chunk = chunk.map(format_to_prompt_completion, remove_columns=chunk.column_names)
    valid_dataset = valid_dataset.map(format_to_prompt_completion, remove_columns=valid_dataset.column_names)

    # Save runtime files
    save_to_jsonl(chunk, "./jsonl/runtime/train.jsonl")
    save_to_jsonl(valid_dataset, "./jsonl/runtime/valid.jsonl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_hf_workspace", type=str, default="aldenirsrv")
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=1000)
    args = parser.parse_args()
    main(args)
