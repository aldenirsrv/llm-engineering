"""
Setup completo do Axolotl para fine-tuning do Llama 3.2

1. Instale as dependências:
   pip install axolotl peft bitsandbytes

2. Prepare o dataset em JSONL com formato correto

3. Configure o arquivo axolotl_config.yaml

4. Rode o treinamento:
   axolotl train axolotl_config.yaml
"""

import json
from pathlib import Path
from transformers import AutoTokenizer

def prepare_dataset_for_axolotl(data, output_path="training_data.jsonl"):
    """
    Prepara dataset no formato correto para Axolotl
    Axolotl espera 'text' com conversas completas no formato chat template
    
    Args:
        data: Lista de dicts com 'instruction', 'input' (opcional), 'output'
        output_path: Caminho para salvar JSONL
    """
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% set loop_messages = messages %}"
            "{% for message in loop_messages %}"
            "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' %}"
            "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
            "{{ content }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
            "{% endif %}"
        )
    
    formatted_data = []
    
    for example in data:
        instruction = example.get("instruction", "")
        inp = example.get("input", "")
        output = example.get("output", "")
        
        # Cria mensagens no formato chat
        messages = [
            {"role": "system", "content": "Você é um assistente útil e preciso."},
            {"role": "user", "content": f"{instruction}\n{inp}".strip()},
            {"role": "assistant", "content": output}
        ]
        
        # Aplica chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Formato para Axolotl: deve ter apenas 'text'
        formatted_data.append({"text": text})
    
    # Salva em JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in formatted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Dataset preparado: {len(formatted_data)} exemplos salvos em {output_path}")
    print(f"Caminho absoluto: {Path(output_path).absolute()}")
    return output_path


def create_axolotl_config(output_path="axolotl_config.yaml"):
    """
    Cria arquivo de configuração do Axolotl
    """
    config_content = """# Axolotl Configuration for Llama 3.2 Fine-tuning

# Model Configuration
base_model: meta-llama/Llama-3.2-3B
model_type: llama
tokenizer_type: llama
trust_remote_code: true

# LoRA Configuration
adapter: lora
lora_model_dir: ./lora_model
use_lora: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - v_proj
  - k_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
lora_fan_in_fan_out: false

# Training Configuration
output_dir: ./axolotl_output
overwrite_output_dir: true
num_train_epochs: 3
per_device_train_batch_size: 2
per_device_eval_batch_size: 2
gradient_accumulation_steps: 2
learning_rate: 5e-5
lr_scheduler_type: cosine
warmup_steps: 100
weight_decay: 0.01
max_grad_norm: 1.0

# Optimization
optim: paged_adamw_32bit
bf16: false
fp16: true
tf32: false
use_gradient_checkpointing: true

# Data Configuration
datasets:
  - path: json
    data_files:
      - ./training_data.jsonl
    type: json
    split: train

val_set_size: 0.1
split_type: random
shuffle_merged_datasets: true

# Sequence Configuration
max_seq_length: 512
pad_token_id: 128002
eos_token_id: 128001
bos_token_id: 128000

# Logging and Checkpointing
logging_steps: 10
save_steps: 50
eval_steps: 50
save_strategy: steps
evaluation_strategy: steps
save_total_limit: 3

# Device and Precision
device_map: auto
dtype: float16

# Misc
seed: 42
mixed_precision: fp16
use_flash_attention_2: false
"""
    
    with open(output_path, 'w') as f:
        f.write(config_content)
    
    print(f"Arquivo de configuração criado: {output_path}")
    return output_path


def create_run_script(output_path="run_training.sh"):
    """
    Cria script para rodar o treinamento
    """
    script_content = """#!/bin/bash

# Script para treinar modelo com Axolotl

echo "===== Instalando dependências ====="
pip install axolotl peft bitsandbytes -q

echo "===== Preparando dataset ====="
python prepare_dataset.py

echo "===== Iniciando treinamento ====="
axolotl train axolotl_config.yaml

echo "===== Treinamento concluído ====="
echo "Modelo salvo em: ./axolotl_output"
echo "LoRA adapters em: ./lora_model"
"""
    
    with open(output_path, 'w') as f:
        f.write(script_content)
    
    print(f"Script criado: {output_path}")
    return output_path


def create_prepare_dataset_script():
    """
    Cria script para preparar o dataset
    """
    script_content = '''"""Prepara dataset para Axolotl"""
import json
from pathlib import Path
from transformers import AutoTokenizer

# Seus dados de treino
data = [
    {
        "instruction": "Responda à pergunta",
        "input": "Qual é a capital do Brasil?",
        "output": "Brasília é a capital do Brasil."
    },
    {
        "instruction": "Traduza para inglês",
        "input": "Olá, como vai?",
        "output": "Hello, how are you?"
    },
    {
        "instruction": "Explique o conceito",
        "input": "O que é inteligência artificial?",
        "output": "Inteligência artificial é a simulação de processos de inteligência humana."
    },
    {
        "instruction": "Responda",
        "input": "Qual é a capital da França?",
        "output": "Paris é a capital da França."
    },
    {
        "instruction": "Traduza",
        "input": "Obrigado",
        "output": "Thank you"
    },
]

# Prepara no formato correto
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

if tokenizer.chat_template is None:
    tokenizer.chat_template = (
        "{% set loop_messages = messages %}"
        "{% for message in loop_messages %}"
        "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\\\n\\\\n'+ message['content'] | trim + '<|eot_id|>' %}"
        "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
        "{{ content }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\\\\n\\\\n' }}"
        "{% endif %}"
    )

with open("training_data.jsonl", "w", encoding="utf-8") as f:
    for example in data:
        instruction = example.get("instruction", "")
        inp = example.get("input", "")
        output = example.get("output", "")
        
        messages = [
            {"role": "system", "content": "Você é um assistente útil."},
            {"role": "user", "content": f"{instruction}\\n{inp}".strip()},
            {"role": "assistant", "content": output}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        f.write(json.dumps({"text": text}, ensure_ascii=False) + "\\n")

print(f"Dataset criado com {len(data)} exemplos")
'''
    
    with open("prepare_dataset.py", 'w') as f:
        f.write(script_content)
    
    print("Script prepare_dataset.py criado")


def main():
    print("===== Setup Axolotl para Fine-tuning =====\n")
    
    # 1. Criar script de preparação
    create_prepare_dataset_script()
    
    # 2. Criar arquivo de configuração
    create_axolotl_config("axolotl_config.yaml")
    
    # 3. Criar script de execução
    create_run_script("run_training.sh")
    
    # 4. Preparar dataset
    data = [
        {
            "instruction": "Responda à pergunta",
            "input": "Qual é a capital do Brasil?",
            "output": "Brasília é a capital do Brasil."
        },
        {
            "instruction": "Traduza para inglês",
            "input": "Olá, como vai?",
            "output": "Hello, how are you?"
        },
        {
            "instruction": "Explique o conceito",
            "input": "O que é inteligência artificial?",
            "output": "Inteligência artificial é a simulação de processos de inteligência humana por computadores."
        },
        {
            "instruction": "Responda",
            "input": "Qual é a capital da França?",
            "output": "Paris é a capital da França."
        },
        {
            "instruction": "Traduza",
            "input": "Obrigado",
            "output": "Thank you"
        },
    ]
    
    prepare_dataset_for_axolotl(data, "training_data.jsonl")
    
    print("\n===== Setup Concluído =====")
    print("\nPróximos passos:")
    print("1. Instale dependências:")
    print("   pip install axolotl peft bitsandbytes")
    print("\n2. Rode o treinamento:")
    print("   axolotl train axolotl_config.yaml")
    print("\nOu use o script:")
    print("   bash run_training.sh")
    print("\nResultados salvos em:")
    print("- ./axolotl_output (modelo e checkpoints)")
    print("- ./lora_model (adapters LoRA)")


if __name__ == "__main__":
    main()