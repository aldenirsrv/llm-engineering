# 1. Delete tudo relacionado
rm axolotl_config.yaml
rm -rf axolotl_output
rm -rf lora_model

# 2. Delete cache do Axolotl
rm -rf ~/.cache/axolotl

# 3. Crie o arquivo manualmente
cat > axolotl_config.yaml << 'EOF'
base_model: meta-llama/Llama-3.2-3B
model_type: llama
tokenizer_type: llama

datasets:
  - path: json
    data_files:
      - ./training_data.jsonl

batch_size: 2
micro_batch_size: 1
gradient_accumulation_steps: 2
num_train_epochs: 2
learning_rate: 1e-4
lr_scheduler_type: linear

adapter: lora
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - v_proj

output_dir: ./axolotl_output
save_strategy: epoch

optim: adamw_torch
fp16: false
bf16: false

max_seq_length: 512
seed: 42
EOF

# 4. Rode
axolotl train axolotl_config.yaml