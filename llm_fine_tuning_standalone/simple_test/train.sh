#!/bin/bash
set -e

echo "🚀 Starting SERIOUS LoRA training on M1 (Llama-3.2-3B)..."

# mlx_lm.lora \
#   --model "meta-llama/Llama-3.2-3B-Instruct" \
#   --train \
#   --data "./jsonl/non-diverse" \
#   --fine-tune-type lora \
#   --num-layers 2 \
#   --batch-size 1 \
#   --max-seq-length 512 \
#   --learning-rate 1e-5 \
#   --iters 100 \
#   --grad-checkpoint \
#   --steps-per-report 5 \
#   --steps-per-eval 20

# ⚡ Sugestão de comando para 8B
mlx_lm.lora \
  --model "meta-llama/Llama-3.2-8B-Instruct" \
  --train \
  --data "./jsonl/non-diverse" \
  --fine-tune-type lora \
  --num-layers 2 \
  --batch-size 1 \
  --max-seq-length 512 \
  --learning-rate 1e-5 \
  --iters 5000 \
  --grad-checkpoint \
  --steps-per-report 10 \
  --steps-per-eval 100

echo "✅ Serious training finished!"