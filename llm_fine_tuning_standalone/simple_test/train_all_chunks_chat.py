import os
import json
import subprocess
from datasets import load_dataset

# =========================
# Config
# =========================
TRAIN_PATH = "./jsonl/non-diverse/train.jsonl"
VALID_PATH = "./jsonl/non-diverse/valid.jsonl"
OUTPUT_DIR = "./jsonl/chunks"
CHUNK_SIZE = 2000  # tamanho de cada pedaço
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_DIR = "adapters"
ITERS_PER_CHUNK = 1000
NUM_LAYERS = 2  # LoRA layers (ajuste para caber no M1)

# =========================
# Split dataset
# =========================
print("📥 Loading dataset...")
dataset = load_dataset("json", data_files=TRAIN_PATH)["train"]
print(f"✅ Loaded {len(dataset)} samples")

os.makedirs(OUTPUT_DIR, exist_ok=True)

chunk_dirs = []

for i in range(0, len(dataset), CHUNK_SIZE):
    chunk = dataset.select(range(i, min(i + CHUNK_SIZE, len(dataset))))
    chunk_dir = os.path.join(OUTPUT_DIR, f"chunk_{i//CHUNK_SIZE}")
    os.makedirs(chunk_dir, exist_ok=True)

    # Save train.jsonl
    train_file = os.path.join(chunk_dir, "train.jsonl")
    with open(train_file, "w", encoding="utf-8") as f:
        for ex in chunk:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Copy valid.jsonl
    valid_file = os.path.join(chunk_dir, "valid.jsonl")
    with open(VALID_PATH, "r", encoding="utf-8") as src, open(valid_file, "w", encoding="utf-8") as dst:
        for line in src:
            dst.write(line)

    chunk_dirs.append(chunk_dir)
    print(f"💾 Saved {len(chunk)} samples → {chunk_dir}")

# =========================
# Train chunk by chunk
# =========================
print("\n🚀 Starting training loop over chunks...")

first_run = True
for chunk_dir in chunk_dirs:
    print(f"\n🔹 Training on {chunk_dir}...")

    if first_run:
        cmd = [
            "mlx_lm.lora",
            "--model", MODEL_NAME,
            "--train",
            "--data", chunk_dir,
            "--fine-tune-type", "lora",
            "--num-layers", str(NUM_LAYERS),
            "--batch-size", "1",
            "--max-seq-length", "512",
            "--learning-rate", "1e-5",
            "--iters", str(ITERS_PER_CHUNK),
            "--grad-checkpoint",
            "--adapter-path", ADAPTER_DIR,
        ]
        first_run = False
    else:
        cmd = [
            "mlx_lm.lora",
            "--model", MODEL_NAME,
            "--train",
            "--data", chunk_dir,
            "--fine-tune-type", "lora",
            "--num-layers", str(NUM_LAYERS),
            "--batch-size", "1",
            "--max-seq-length", "512",
            "--learning-rate", "1e-5",
            "--iters", str(ITERS_PER_CHUNK),
            "--grad-checkpoint",
            "--resume-adapter-file", f"{ADAPTER_DIR}/adapters.safetensors",
            "--adapter-path", ADAPTER_DIR,
        ]

    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

print("\n🎉 Training completed on all chunks!")