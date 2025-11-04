import argparse
import os
import yaml
from datasets import load_dataset, concatenate_datasets, Dataset
from typing import List, Literal, TypedDict, NamedTuple, Tuple, Dict, Union, Optional
from pydantic import BaseModel

# Import comet early
try:
    import comet_ml
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False

from transformers import AutoTokenizer
from huggingface_hub import HfApi, HfFolder, Repository, create_repo

from mlx_lm import load
from mlx_lm.tuner import TrainingArgs, datasets, train
from mlx_lm.tuner.lora import LoRALinear
import mlx.optimizers as optim
import mlx.nn as nn
import mlx.core as mx
from dotenv import load_dotenv

load_dotenv()


# ==========================================
# Data Models
# ==========================================
class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class FormattedExample(TypedDict):
    messages: List[Message]

class SFTDatasets(NamedTuple):
    train: Dataset
    test: Dataset


# ==========================================
# Métricas
# ==========================================
class Metrics:
    def __init__(self) -> None:
        self.train_losses: List[Tuple[int, float]] = []
        self.val_losses: List[Tuple[int, float]] = []

    def on_train_loss_report(self, info: Dict[str, Union[float, int]]) -> None:
        self.train_losses.append((info["iteration"], info["train_loss"]))

    def on_val_loss_report(self, info: Dict[str, Union[float, int]]) -> None:
        self.val_losses.append((info["iteration"], info["val_loss"]))


# ==========================================
# Conversão do dataset
# ==========================================
def format_to_chat(example) -> FormattedExample:
    """Transforma instruction/output em lista de mensagens (para debug/export)"""
    system_content = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    
    messages: List[Message] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": example['instruction']},
        {"role": "assistant", "content": example['output']}
    ]
    
    return {"messages": messages}


def to_prompt_completion(example):
    """Transforma messages em prompt/completion"""
    user_msgs = [m["content"] for m in example["messages"] if m["role"] == "user"]
    assistant_msgs = [m["content"] for m in example["messages"] if m["role"] == "assistant"]

    prompt = "\n".join([f"User: {m}" for m in user_msgs])
    completion = "\n".join([f"Assistant: {m}" for m in assistant_msgs])

    return {"prompt": prompt, "completion": completion}


# ==========================================
# LoRA injection
# ==========================================
def load_model(model_name: str, target_modules: List[str], lora_rank=8, dropout=0.05, scale=16):
    model, tokenizer = load(model_name)  # Tokenizer aqui é wrapper do MLX
    print(f"✅ Loaded base model: {model_name}")
    visited = set()

    def apply_lora(module, prefix=""):
        if id(module) in visited: return
        visited.add(id(module))
        ignore = {"parent", "model", "modules", "weights", "training"}
        for name in dir(module):
            if name.startswith("_") or name in ignore: continue
            try:
                child = getattr(module, name)
            except Exception:
                continue
            if isinstance(child, (nn.Linear, nn.QuantizedLinear)) and any(t in name for t in target_modules):
                setattr(module, name, LoRALinear.from_linear(child, r=lora_rank, alpha=scale, dropout=dropout))
                print(f"💡 Added LoRA: {prefix}.{name}")
            elif isinstance(child, list):
                for i, sub in enumerate(child):
                    if hasattr(sub, "__dict__"):
                        apply_lora(sub, f"{prefix}.{name}[{i}]")
            elif hasattr(child, "__dict__"):
                apply_lora(child, f"{prefix}.{name}" if prefix else name)

    apply_lora(model)
    print("✅ LoRA injection complete.")
    return model, tokenizer


# ==========================================
# Utils
# ==========================================
def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    

def prepare_and_export_sft(dataset_hf_workspace: str, is_dummy: bool = False) -> SFTDatasets:
    dataset1 = load_dataset(f"{dataset_hf_workspace}/llmtwin", split="train")
    dataset2 = load_dataset("mlabonne/FineTome-Alpaca-100k", split="train[:10000]")
    dataset = concatenate_datasets([dataset1, dataset2])
    
    if is_dummy:
        dataset = dataset.select(range(min(100, len(dataset))))
    
    print(f"✅ Loaded dataset with {len(dataset)} samples.")

    dataset = dataset.train_test_split(test_size=0.05)

    train_dataset = dataset["train"].map(
        format_to_chat, remove_columns=dataset["train"].column_names
    )
    test_dataset = dataset["test"].map(
        format_to_chat, remove_columns=dataset["test"].column_names
    )
    return train_dataset, test_dataset


def save_to_jsonl(dataset: Dataset, filename: str):
    import json
    with open(filename, 'w', encoding='utf-8') as f:
        for example in dataset:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    print(f"💾 Salvo: {filename}")


# ==========================================
# SFT Training
# ==========================================
def run_sft(config_path: str, output_repo_id: Optional[str] = None):
    cfg = load_config(config_path)
    metrics = Metrics()

    # Carrega modelo + LoRA
    model, _ = load_model(cfg["model"], cfg["lora_parameters"]["keys"],
                          lora_rank=cfg["lora_parameters"]["rank"],
                          dropout=cfg["lora_parameters"]["dropout"],
                          scale=cfg["lora_parameters"]["alpha"])

    # Tokenizer Hugging Face
    hf_tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    hf_tokenizer.pad_token = hf_tokenizer.eos_token

    # Prepara dataset
    train_dataset, test_dataset = prepare_and_export_sft(dataset_hf_workspace=args.dataset_hf_workspace, is_dummy=args.is_dummy)

    # Converte para prompt/completion
    train_ready = train_dataset.map(to_prompt_completion, remove_columns=train_dataset.column_names)
    test_ready = test_dataset.map(to_prompt_completion, remove_columns=test_dataset.column_names)

    # Debug: salvar JSONL
    os.makedirs("data_see", exist_ok=True)
    save_to_jsonl(train_ready, "data_see/train_prompt_completion.jsonl")
    save_to_jsonl(test_ready, "data_see/test_prompt_completion.jsonl")

    print(f"✅ Train: {len(train_ready)} exemplos")
    print(f"✅ Test: {len(test_ready)} exemplos")
    print(f"Exemplo:\n{train_ready[0]}")

    # Converte para lista de dicionários (formato suportado pelo mlx-lm)
    train_dicts = [{"prompt": ex["prompt"], "completion": ex["completion"]} for ex in train_ready]
    val_dicts = [{"prompt": ex["prompt"], "completion": ex["completion"]} for ex in test_ready]

    # Cria datasets no formato mlx
    configs = {
        "mask_prompt": False,
        "prompt_feature": "prompt",
        "completion_feature": "completion",
    }
    train_set = datasets.create_dataset(train_dicts, hf_tokenizer, configs)
    val_set = datasets.create_dataset(val_dicts, hf_tokenizer, configs)

    # Treino
    training_args = TrainingArgs(**{k: v for k, v in cfg.items() if k in TrainingArgs.__annotations__})
    print("🚀 Starting SFT training...")
    train(
        model=model, 
        train_dataset=train_set,
        val_dataset=val_set,
        args=training_args, 
        training_callback=metrics,
        optimizer=optim.Adam(learning_rate=1e-5)
    )


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_dummy", type=bool, default=False)
    parser.add_argument("--dataset_hf_workspace", type=str, default="aldenirsrv")
    parser.add_argument("--config_path", type=str, default="sft_config.yml")
    parser.add_argument("--model_output_huggingface_workspace", type=str, default="aldenirsrv")
    args = parser.parse_args()

    sft_output_model_repo_id = f"{args.model_output_huggingface_workspace}/TwinLlama-3.1-8B"
    run_sft(config_path=args.config_path, output_repo_id=sft_output_model_repo_id)