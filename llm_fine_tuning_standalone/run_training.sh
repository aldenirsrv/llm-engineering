#!/bin/bash

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
