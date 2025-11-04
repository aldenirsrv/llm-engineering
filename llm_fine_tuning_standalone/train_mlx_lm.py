# ==========================================================
# MLX-LM Fine-tuning Script (SFT + DPO + YAML + Comet ML + HuggingFace Upload)
# ==========================================================
import os
import argparse
import yaml
from typing import List, Optional, Tuple, Union, Dict
import pandas as pd

# Import comet early
try:
    import comet_ml
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False

from datasets import load_dataset, concatenate_datasets
from huggingface_hub import HfApi, HfFolder, Repository, create_repo
from mlx_lm import load, generate
from mlx_lm.tuner import TrainingArgs, datasets, linear_to_lora_layers, train
from mlx_lm.tuner.lora import LoRALinear
import mlx.optimizers as optim
import mlx.nn as nn
import mlx.core as mx
from dotenv import load_dotenv


load_dotenv()



# ==========================================
# Helpers
# ==========================================

def format_sft_samples(examples, eos_token):
    """Format Chat-style SFT samples."""
    texts = [
        alpaca_template.format(inst, out) + eos_token
        for inst, out in zip(examples["instruction"], examples["output"])
    ]
    return {"text": texts}


# def format_dpo_samples(example, eos_token):
#     """Format DPO samples."""
#     example["prompt"] = alpaca_template.format(example["prompt"], "")
#     example["chosen"] = example["chosen"] + eos_token
#     example["rejected"] = example["rejected"] + eos_token
#     return example




# ==========================================
# Comet ML
# ==========================================
def init_comet(enabled: bool = False, project_name: str = "mlx-lm-finetuning"):
    if not enabled or not COMET_AVAILABLE:
        return None
    api_key = os.getenv("COMET_API_KEY")
    if not api_key:
        print("⚠️ COMET_API_KEY not found. Skipping Comet logging.")
        return None
    comet_ml.start(
    project_name="my_project",
    api_key=api_key,
    experiment_config=comet_ml.ExperimentConfig(
        name="my_experiment",
        
        tags=["my_tag"],
        parse_args=False,
    ),
)
    exp = comet_ml.Experiment(api_key=api_key, project_name=project_name)
    print(f"📡 Comet ML enabled — project: {project_name}")
    return exp

class Metrics:
    def __init__(self) -> None:
        self.train_losses: List[Tuple[int, float]] = []
        self.val_losses: List[Tuple[int, float]] = []

    def on_train_loss_report(self, info: Dict[str, Union[float, int]]) -> None:
        self.train_losses.append((info["iteration"], info["train_loss"]))

    def on_val_loss_report(self, info: Dict[str, Union[float, int]]) -> None:
        self.val_losses.append((info["iteration"], info["val_loss"]))
        
# ==========================================
# YAML loader
# ==========================================
def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ==========================================
# LoRA injection
# ==========================================
def load_model(model_name: str, target_modules: List[str], lora_rank=8, dropout=0.05, scale=16):
    model, tokenizer = load(model_name)
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
# Dataset Export (HF → JSONL for MLX)
# ==========================================
def prepare_and_export_sft(dataset_hf_workspace: str, is_dummy: bool = False):
    ## DATASETS TO TRAINING ##
    dataset1 = load_dataset(f"{dataset_hf_workspace}/llmtwin", split="train")
    dataset2 = load_dataset("mlabonne/FineTome-Alpaca-100k", split="train[:10000]")

    ## ALL DATA CONCATENATED ##
    dataset = concatenate_datasets([dataset1, dataset2])
    pandas_df_test = pd.DataFrame(dataset)
    print(pandas_df_test)
    
    if is_dummy:
        dataset = dataset.select(range(400))
    
    ### ALPACA TEMPLATE ###
    dataset = dataset.map(
        lambda ex: {
            "text": f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}"
        },
        remove_columns=dataset.column_names
    )
    dataset = dataset.train_test_split(test_size=0.05)
    os.makedirs("data", exist_ok=True)
    dataset["train"].to_json("data/train.jsonl")
    dataset["test"].to_json("data/valid.jsonl")
    print(f"📊 Exported Train={len(dataset['train'])}, Valid={len(dataset['test'])}")
    train_data =  dataset['train']
    test_data =   dataset['test']
    return train_data, test_data

def prepare_dpo_dataset(dataset_hf_workspace: str, is_dummy: bool = False):
    dataset = load_dataset(f"{dataset_hf_workspace}/llmtwin-dpo", split="train")
    if is_dummy: dataset = dataset.select(range(400))
    dataset = dataset.map(lambda ex: {
        "prompt": f"Below is an instruction...\n\n### Instruction:\n{ex['prompt']}\n\n### Response:\n",
        "chosen": ex["chosen"], "rejected": ex["rejected"]
    }, remove_columns=dataset.column_names)
    return dataset

# ==========================================
# Save + Quantize + Push to HF
# ==========================================
def save_and_quantize(model, tokenizer, output_dir: str, bits: int = 4):
    from mlx_lm.quantization import quantize
    qmodel = quantize(model, bits=bits)
    qmodel.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"💾 Saved quantized model to {output_dir}")
    return qmodel

def save_and_push_model(model, tokenizer, output_dir: str, repo_id: str, private: bool = True):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    if repo_id:
        try:
            print(f"🚀 Pushing to HF Hub: {repo_id}")
            token = HfFolder.get_token()
            if not token: raise RuntimeError("No HF token. Run `huggingface-cli login`.")
            create_repo(repo_id, exist_ok=True, private=private)
            repo = Repository(local_dir=output_dir, clone_from=repo_id, use_auth_token=token)
            repo.push_to_hub(commit_message="Upload fine-tuned MLX model")
            print(f"✅ Model pushed to {repo_id}")
        except Exception as e:
            print(f"⚠️ HF push failed: {e}")

# ==========================================
# SFT Training (MLX-native, YAML-driven)
# ==========================================
def run_sft(config_path: str, report_to_comet: bool = False, repo_id: Optional[str] = None):
    cfg = load_config(config_path)
    metrics = Metrics()
    model, tokenizer = load_model(cfg["model"], cfg["lora_parameters"]["keys"],
                                  lora_rank=cfg["lora_parameters"]["rank"],
                                  dropout=cfg["lora_parameters"]["dropout"],
                                  scale=cfg["lora_parameters"]["alpha"])
    train_data, test_data = prepare_and_export_sft(cfg.get("dataset_hf_workspace", "mlabonne"), cfg.get("is_dummy", False))

    print('### DATA SET READY TO TRAIN')
    print(train_data)
    # Convert just the train split
    pandas_df = pd.DataFrame(train_data)

    print(pandas_df)

    # Or convert the test split
    pandas_df_test = pd.DataFrame(test_data)
    print(pandas_df_test)

    # Convert to tuple format for MLX-LM
    train_data_mlx = [(item['text'],) for item in train_data]
    test_data_mlx = [(item['text'],) for item in test_data]
    
   
    exp = init_comet(enabled=report_to_comet)
    training_args = TrainingArgs(**{k: v for k, v in cfg.items() if k in TrainingArgs.__annotations__})

    # def comet_callback(step, metrics):
    #     if exp:
    #         for k, v in metrics.items():
    #             exp.log_metric(k, v, step=step)

    configs = {
        "mask_prompt": False,
        "prompt_feature": "prompt",
        "text_feature": "text",
        "completion_feature": "completion",
        "chat_feature": "messages",
    }
    train_set = datasets.create_dataset(
        train_data_mlx,
        tokenizer,
        configs
    )
    val_set = datasets.create_dataset(
        test_data_mlx,
        tokenizer,
        configs
    )

    print("🚀 Starting SFT training...")
    train(
        model=model, 
        train_dataset=train_set,
        val_dataset=val_set,
        args=training_args, 
        training_callback=metrics,
        optimizer=optim.Adam(learning_rate=1e-5)
    )

    qmodel = save_and_quantize(model, tokenizer, cfg.get("adapter_file", "output_sft"), bits=cfg.get("quant_bits", 4))
    save_and_push_model(qmodel, tokenizer, cfg.get("adapter_file", "output_sft"), repo_id)
    if exp: exp.end()
    return qmodel, tokenizer

# ==========================================
# DPO Training (Manual)
# ==========================================
def compute_logprob(model, tokenizer, prompt: str, response: str):
    inputs = tokenizer(prompt + response, return_tensors="pt")
    tokens = mx.array(inputs["input_ids"][0])
    logits = model(tokens[:-1][None, :])
    probs = nn.log_softmax(logits, axis=-1)
    log_probs = mx.take_along_axis(probs, tokens[1:][None, :, None], axis=-1).squeeze(-1)
    return mx.mean(log_probs).item()

def run_dpo(config_path: str, report_to_comet: bool = False, repo_id: Optional[str] = None):
    cfg = load_config(config_path)
    model, tokenizer = load_model(cfg["model"], cfg["lora_parameters"]["keys"],
                                  lora_rank=cfg["lora_parameters"]["rank"],
                                  dropout=cfg["lora_parameters"]["dropout"],
                                  scale=cfg["lora_parameters"]["alpha"])
    dataset = prepare_dpo_dataset(cfg.get("dataset_hf_workspace", "mlabonne"), cfg.get("is_dummy", False))
    exp = init_comet(enabled=report_to_comet)

    optimizer = mx.optimizers.AdamW(model.parameters(), lr=cfg.get("learning_rate", 2e-6))
    beta = cfg.get("beta", 0.5)

    for epoch in range(cfg.get("num_train_epochs", 1)):
        total_loss = 0.0
        for ex in dataset:
            logp_chosen = compute_logprob(model, tokenizer, ex["prompt"], ex["chosen"])
            logp_rejected = compute_logprob(model, tokenizer, ex["prompt"], ex["rejected"])
            loss = -mx.log(mx.sigmoid(beta * (logp_chosen - logp_rejected)))
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1} | Avg DPO Loss: {avg_loss:.4f}")
        if exp: exp.log_metric("epoch_loss", avg_loss, epoch=epoch)

    qmodel = save_and_quantize(model, tokenizer, cfg.get("adapter_file", "output_dpo"), bits=cfg.get("quant_bits", 4))
    save_and_push_model(qmodel, tokenizer, cfg.get("adapter_file", "output_dpo"), repo_id)
    if exp: exp.end()
    return qmodel, tokenizer

# ==========================================
# CLI
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--report_to", type=str, default=None)
    parser.add_argument("--repo_id", type=str, help="HF repo id")
    args = parser.parse_args()

    report_to_comet = args.report_to == "comet_ml"
    cfg = load_config(args.config)

    if cfg.get("finetuning_type", "sft") == "sft":
        run_sft(args.config, report_to_comet, repo_id=args.repo_id)
    else:
        run_dpo(args.config, report_to_comet, repo_id=args.repo_id)