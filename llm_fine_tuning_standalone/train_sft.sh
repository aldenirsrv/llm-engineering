#!/usr/bin/env bash
set -e

# Activate venv
source .venv/bin/activate

# Hugging Face auth check
if ! huggingface-cli whoami > /dev/null 2>&1; then
  echo "⚠️  Not logged in to Hugging Face. Please login:"
  huggingface-cli login
else
  echo "✅ Already logged in to Hugging Face."
fi

# Debug info
echo "🚀 Starting fine-tuning"
echo "   Base model: meta-llama/Llama-3.1-8B"
echo "   Seq length: 2048"
echo "   Batch size: 2"
echo "   Epochs: 3"
echo "   LR: 3e-4"
echo "   LoRA rank: 32"

# Run fine-tuning
python -m src.main.main \
  --finetuning_type sft \
  --base_model_name meta-llama/Llama-3.1-8B \
  --max_seq_length 2048 \
  --per_device_train_batch_size 2 \
  --num_train_epochs 3 \
  --learning_rate 3e-4 \
  --lora_rank 32 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --chat_template chatml \
  --sft_datasets aldenirsrv/llmtwin:train mlabonne/FineTome-Alpaca-100k:train[:10000] \
  --model_output_huggingface_workspace aldenirsrv \
  --output_data_dir outputs \
  --model_dir models \
  --n_gpus 1 \
  --load_in_4bit