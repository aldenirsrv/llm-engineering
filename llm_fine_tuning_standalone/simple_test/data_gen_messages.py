import os
import re
import argparse
import json
from datasets import load_dataset, concatenate_datasets, Dataset
from typing import List, Literal, TypedDict, NamedTuple, Tuple, Dict, Union, Optional
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

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")\
# ==========================================
# Dataset conversion
# ==========================================

def clean_text(text: str) -> str:
    # Replace Unicode line/paragraph separators with normal newlines
    return re.sub(r'[\u2028\u2029]', '\n', text)

def deep_clean(obj):
    """Recursively clean strings inside dicts/lists."""
    if isinstance(obj, str):
        # Replace unusual unicode separators with normal newline
        return re.sub(r'[\u2028\u2029]', '\n', obj)
    elif isinstance(obj, dict):
        return {k: deep_clean(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_clean(i) for i in obj]
    else:
        return obj
# def format_to_chat(example):
#     """Convert instruction/output into prompt and completion."""
#     return {
#         "prompt": example["instruction"],
#         "completion": example["output"],
#     }

# ==========================================
# Conversão do dataset
# ==========================================
def format_to_chat(example):
    """Convert instruction/output into chat-style messages."""
    system_content = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": example['instruction']},
        {"role": "assistant", "content": example['output']}
    ]
    
    return {"messages": messages}

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
    print(dataset)

    # Map to prompt/completion
    train_dataset = dataset["train"].map(
        format_to_chat, remove_columns=dataset["train"].column_names
    )
    test_dataset = dataset["test"].map(
        format_to_chat, remove_columns=dataset["test"].column_names
    )

    # Export to JSONL
    save_to_jsonl(train_dataset, "./jsonl/non-diverse/train.jsonl")
    save_to_jsonl(test_dataset, "./jsonl/non-diverse/valid.jsonl")

    return train_dataset, test_dataset


def save_to_jsonl(dataset: Dataset, filename: str):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w', encoding='utf-8', newline='\n') as f:
        for example in dataset:
            # Recursively clean all string values
            cleaned = {
                k: deep_clean(v) if isinstance(v, str) else v
                for k, v in example.items()
            }
            f.write(json.dumps(cleaned, ensure_ascii=False) + '\n')
    print(f"💾 Saved: {filename}")

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_dummy", type=bool, default=False)
    parser.add_argument("--dataset_hf_workspace", type=str, default="aldenirsrv")
    args = parser.parse_args()

    prepare_and_export_sft(dataset_hf_workspace=args.dataset_hf_workspace, is_dummy=args.is_dummy)