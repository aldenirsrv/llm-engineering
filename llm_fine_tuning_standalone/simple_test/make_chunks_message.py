import os
import json
from datasets import load_dataset

# 📂 Estrutura final

# Depois de rodar python make_chunks.py, você vai ter:

# jsonl/chunks/
#  ├── chunk_0/
#  │   ├── train.jsonl   (2000 exemplos)
#  │   └── valid.jsonl   (mesmo arquivo de validação)
#  ├── chunk_1/
#  │   ├── train.jsonl
#  │   └── valid.jsonl
#  ...
# Config
TRAIN_PATH = "./jsonl/non-diverse/train.jsonl"
VALID_PATH = "./jsonl/non-diverse/valid.jsonl"
OUTPUT_DIR = "./jsonl/non-diverse/message/chunks"
CHUNK_SIZE = 1000  # você pode trocar para 1000 se quiser menor

# Load dataset
print("📥 Loading dataset...")
dataset = load_dataset("json", data_files=TRAIN_PATH)["train"]
print(f"✅ Loaded {len(dataset)} samples")

# Create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Split into chunks
for i in range(0, len(dataset), CHUNK_SIZE):
    chunk = dataset.select(range(i, min(i + CHUNK_SIZE, len(dataset))))
    chunk_dir = os.path.join(OUTPUT_DIR, f"chunk_{i//CHUNK_SIZE}")
    os.makedirs(chunk_dir, exist_ok=True)

    # Save train.jsonl
    train_file = os.path.join(chunk_dir, "train.jsonl")
    with open(train_file, "w", encoding="utf-8") as f:
        for ex in chunk:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Copy valid.jsonl into each chunk dir
    valid_file = os.path.join(chunk_dir, "valid.jsonl")
    with open(VALID_PATH, "r", encoding="utf-8") as src, open(valid_file, "w", encoding="utf-8") as dst:
        for line in src:
            dst.write(line)

    print(f"💾 Saved {len(chunk)} samples → {chunk_dir}")

print("🎉 All chunks created!")
