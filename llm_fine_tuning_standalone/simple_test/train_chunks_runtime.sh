#!/bin/bash
set -e

MODEL="meta-llama/Llama-3.2-3B"
ADAPTER_DIR="adapters-docwriter"

ITERS=10
BATCH_SIZE=1
SEQ_LEN=512
LR=1e-5
LAYERS=4
CHUNK_SIZE=1000
MAX_CHUNKS=${MAX_CHUNKS:-0} # 0 = train all

mkdir -p "$ADAPTER_DIR"
count=0
PREV_ADAPTER=""

while true; do
    if [ "$MAX_CHUNKS" -gt 0 ] && [ "$count" -ge "$MAX_CHUNKS" ]; then
        echo "⏹️ Stopping after $MAX_CHUNKS chunks."
        break
    fi

    echo "📦 Preparing chunk $count ..."
    python prepare_chunk_runtime.py --chunk-id "$count" --chunk-size "$CHUNK_SIZE"

    # If no data generated, stop
    if [ ! -s "./jsonl/runtime/train.jsonl" ]; then
        echo "✅ No more chunks to process."
        break
    fi

    echo "🚀 Training on chunk $count ..."

    CMD="mlx_lm.lora \
      --model $MODEL \
      --train \
      --data ./jsonl/runtime \
      --fine-tune-type lora \
      --num-layers $LAYERS \
      --batch-size $BATCH_SIZE \
      --max-seq-length $SEQ_LEN \
      --learning-rate $LR \
      --iters $ITERS \
      --grad-checkpoint \
      --adapter-path $ADAPTER_DIR"

    # Resume if not the first chunk
    if [ -n "$PREV_ADAPTER" ]; then
        CMD="$CMD --resume-adapter-file $PREV_ADAPTER"
    fi

    echo "🔹 Running: $CMD"
    eval $CMD

    # Save adapter incrementally
    CHUNK_ID=$(printf "%07d" $(( (count + 1) * ITERS )))
    SAVED_ADAPTER="$ADAPTER_DIR/${CHUNK_ID}_adapters.safetensors"
    cp "$ADAPTER_DIR/adapters.safetensors" "$SAVED_ADAPTER"
    echo "💾 Saved adapter: $SAVED_ADAPTER"

    # Set for next chunk
    PREV_ADAPTER="$SAVED_ADAPTER"
    count=$((count + 1))
done

echo "🎉 Finished training on $count chunks!"