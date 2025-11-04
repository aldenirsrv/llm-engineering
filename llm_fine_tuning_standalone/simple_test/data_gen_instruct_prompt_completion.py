import os
import re
import argparse
import json
from datasets import load_dataset, concatenate_datasets, Dataset
from typing import List, Literal, TypedDict
from pydantic import BaseModel
from transformers import AutoTokenizer

# ==========================================
# Data Models
# ==========================================
class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class FormattedExample(TypedDict):
    messages: List[Message]

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

CHUNK_SIZE = 1000

# ==========================================
# Cleaning utils
# ==========================================
def deep_clean(obj):
    """Recursively clean strings inside dicts/lists."""
    if isinstance(obj, str):
        return re.sub(r'[\u2028\u2029]', '\n', obj)
    elif isinstance(obj, dict):
        return {k: deep_clean(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_clean(i) for i in obj]
    else:
        return obj

# ==========================================
# Dataset conversion
# ==========================================
def format_to_prompt_completion(example):
    """Convert instruction/output into prompt/completion."""
    return {
        "prompt": example["instruction"],
        "completion": example["output"],
    }

# (opcional) manter chat-style para debug
def format_to_chat(example):
    system_content = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": example['instruction']},
        {"role": "assistant", "content": example['output']}
    ]
    return {"messages": messages}

# ==========================================
# Save JSONL
# ==========================================
def save_to_jsonl(dataset: Dataset, filename: str):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8', newline='\n') as f:
        for example in dataset:
            cleaned = deep_clean(example)
            f.write(json.dumps(cleaned, ensure_ascii=False) + '\n')
    print(f"💾 Saved {len(dataset)} → {filename}")

# ==========================================
# Prepare datasets
# ==========================================
def prepare_and_export_sft(dataset_hf_workspace: str, is_dummy: bool = False):
    # Load datasets
    dataset1 = load_dataset(f"{dataset_hf_workspace}/llmtwin", split="train")
    dataset2 = load_dataset("mlabonne/FineTome-Alpaca-100k", split="train[:10000]")
    dataset = concatenate_datasets([dataset1, dataset2])

    if is_dummy:
        dataset = dataset.select(range(min(100, len(dataset))))

    print(f"✅ Loaded dataset with {len(dataset)} samples.")

    # Train/test split
    dataset = dataset.train_test_split(test_size=0.05)
    train_dataset = dataset["train"].map(format_to_prompt_completion, remove_columns=dataset["train"].column_names)
    valid_dataset = dataset["test"].map(format_to_prompt_completion, remove_columns=dataset["test"].column_names)

    # Save full train/valid
    save_to_jsonl(train_dataset, "./jsonl/non-diverse/train.jsonl")
    save_to_jsonl(valid_dataset, "./jsonl/non-diverse/valid.jsonl")

    # Save chunks of 1000
    for i in range(0, len(train_dataset), CHUNK_SIZE):
        chunk = train_dataset.select(range(i, min(i + CHUNK_SIZE, len(train_dataset))))
        chunk_dir = f"./jsonl/chunks/chunk_{i//CHUNK_SIZE}"
        os.makedirs(chunk_dir, exist_ok=True)
        save_to_jsonl(chunk, os.path.join(chunk_dir, "train.jsonl"))
        save_to_jsonl(valid_dataset, os.path.join(chunk_dir, "valid.jsonl"))

    return train_dataset, valid_dataset

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_dummy", type=bool, default=False)
    parser.add_argument("--dataset_hf_workspace", type=str, default="aldenirsrv")
    args = parser.parse_args()

    prepare_and_export_sft(dataset_hf_workspace=args.dataset_hf_workspace, is_dummy=args.is_dummy)


#     📂 Estrutura gerada
# jsonl/
#  ├── non-diverse/
#  │   ├── train.jsonl
#  │   └── valid.jsonl
#  └── chunks/
#      ├── chunk_0/train.jsonl
#      ├── chunk_0/valid.jsonl
#      ├── chunk_1/train.jsonl
#      ├── chunk_1/valid.jsonl
#      ...