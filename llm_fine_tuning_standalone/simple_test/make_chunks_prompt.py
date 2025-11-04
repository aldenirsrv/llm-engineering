import os
import json
from datasets import load_dataset, concatenate_datasets

CHUNK_SIZE = 1000

def format_to_chat(example):
    """Convert instruction/output into prompt/completion"""
    return {
        "prompt": example["instruction"],
        "completion": example["output"],
    }

def save_to_jsonl(dataset, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for ex in dataset:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"💾 Saved {len(dataset)} → {filepath}")

def prepare_and_export_sft(dataset_hf_workspace: str, is_dummy: bool = False):
    # Load datasets from Hugging Face
    dataset1 = load_dataset(f"{dataset_hf_workspace}/llmtwin", split="train")
    dataset2 = load_dataset("mlabonne/FineTome-Alpaca-100k", split="train[:10000]")
    dataset = concatenate_datasets([dataset1, dataset2])
    
    if is_dummy:
        dataset = dataset.select(range(min(100, len(dataset))))
    
    print(f"✅ Loaded dataset with {len(dataset)} samples.")

    # Train/test split
    dataset = dataset.train_test_split(test_size=0.05)
    train_dataset = dataset["train"].map(format_to_chat, remove_columns=dataset["train"].column_names)
    valid_dataset = dataset["test"].map(format_to_chat, remove_columns=dataset["test"].column_names)

    # Save full train/valid
    save_to_jsonl(train_dataset, "./jsonl/non-diverse/train.jsonl")
    save_to_jsonl(valid_dataset, "./jsonl/non-diverse/valid.jsonl")

    # Optionally: save train chunks of 1000
    for i in range(0, len(train_dataset), CHUNK_SIZE):
        chunk = train_dataset.select(range(i, min(i+CHUNK_SIZE, len(train_dataset))))
        chunk_dir = f"./jsonl/chunks/chunk_{i//CHUNK_SIZE}"
        os.makedirs(chunk_dir, exist_ok=True)
        save_to_jsonl(chunk, os.path.join(chunk_dir, "train.jsonl"))
        save_to_jsonl(valid_dataset, os.path.join(chunk_dir, "valid.jsonl"))

    return train_dataset, valid_dataset