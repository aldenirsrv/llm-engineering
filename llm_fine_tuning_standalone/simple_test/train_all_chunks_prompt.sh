#!/bin/bash
set -e

# ==============================
# Config
# ==============================
MODEL="meta-llama/Llama-3.1-8B"
CHUNKS_DIR="./jsonl/chunks"
ADAPTER_DIR="adapters-docwriter"

# Hyperparameters
ITERS=10
BATCH_SIZE=1
SEQ_LEN=512
LR=1e-5
LAYERS=4

# ==============================
# Optional: limit chunks
# Pass MAX_CHUNKS as env var or argument
# Example: MAX_CHUNKS=2 ./train_chunks.sh
# ==============================
MAX_CHUNKS=${MAX_CHUNKS:-0} # 0 = train all chunks

# ==============================
# Training loop
# ==============================
first_chunk=true
count=0

for chunk in $(ls -d $CHUNKS_DIR/chunk_* | sort -V); do
    # If MAX_CHUNKS > 0, stop after that many
    if [ "$MAX_CHUNKS" -gt 0 ] && [ "$count" -ge "$MAX_CHUNKS" ]; then
        echo "⏹️ Stopping after $MAX_CHUNKS chunks."
        break
    fi

    echo "🚀 Training on $chunk ..."

    CMD="mlx_lm.lora \
      --model $MODEL \
      --train \
      --data $chunk \
      --fine-tune-type lora \
      --num-layers $LAYERS \
      --batch-size $BATCH_SIZE \
      --max-seq-length $SEQ_LEN \
      --learning-rate $LR \
      --iters $ITERS \
      --grad-checkpoint \
      --prompt-feature prompt \
      --completion-feature completion \
      --adapter-path $ADAPTER_DIR"

    if [ "$first_chunk" = false ]; then
        CMD="$CMD --resume-adapter-file $ADAPTER_DIR/adapters.safetensors"
    fi

    echo "🔹 Running: $CMD"
    eval $CMD

    first_chunk=false
    count=$((count + 1))
done

echo "🎉 Finished training on $count chunks!"