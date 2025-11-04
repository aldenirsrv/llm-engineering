#!/bin/bash
set -e

# ==============================
# Config
# ==============================
MODEL="meta-llama/Llama-3.2-3B-Instruct"
CHUNKS_DIR="./jsonl/non-diverse/message/chunks"
ADAPTER_DIR="adapters-docwriter-rag"

# Hyperparameters
ITERS=100
BATCH_SIZE=4
SEQ_LEN=1024
LR=1e-5
LAYERS=8 #8
STEPS_PER_REPORT=25
STEPS_PER_EVAL=300
GRADIENT_ACCUM_STEPS=8    # or more if BATCH_SIZE must stay at 1

# lora_config = {
#     "num_layers": 8,
#     "lora_parameters": {
#         "rank": 8,
#         "scale": 20.0,
#         "dropout": 0.0,
#     },
# }

echo "🚀 Training model:  $MODEL"
mkdir -p "$ADAPTER_DIR"  # Garante que o diretório existe

PREV_ADAPTER=""

for chunk_dir in "$CHUNKS_DIR"/chunk_*; do
  if [ ! -d "$chunk_dir" ]; then
    echo "⚠️  Pasta de chunk não encontrada: $chunk_dir"
    continue
  fi

  STEP_NUM=$(basename "$chunk_dir" | grep -o '[0-9]*' | xargs printf "%07d")
  echo "🚀 Treinando no chunk $chunk_dir (step $STEP_NUM)"

  CMD="mlx_lm.lora \
    --model \"$MODEL\" \
    --train \
    --data \"$chunk_dir\" \
    --fine-tune-type lora \
    --num-layers $LAYERS \
    --batch-size $BATCH_SIZE \
    --max-seq-length $SEQ_LEN \
    --learning-rate $LR \
    --iters $ITERS \
    --optimizer adamw \
    --grad-checkpoint \
    --adapter-path \"$ADAPTER_DIR\""

  if [ -n "$PREV_ADAPTER" ]; then
    CMD="$CMD --resume-adapter-file \"$PREV_ADAPTER\""
  fi

  echo "🔹 Executando comando:"
  echo "$CMD"
  eval $CMD

  # Salva o adapter incrementalmente
  SAVED_ADAPTER="$ADAPTER_DIR/${STEP_NUM}_adapters.safetensors"
  cp "$ADAPTER_DIR/adapters.safetensors" "$SAVED_ADAPTER"
  echo "💾 Adapter salvo em: $SAVED_ADAPTER"

  # Define o adapter para o próximo chunk
  PREV_ADAPTER="$SAVED_ADAPTER"
done

echo "✅ Treinamento concluído em todos os chunks!"