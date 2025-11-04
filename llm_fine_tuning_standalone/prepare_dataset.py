"""Prepara dataset para Axolotl"""
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
        "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' %}"
        "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
        "{{ content }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
        "{% endif %}"
    )

with open("training_data.jsonl", "w", encoding="utf-8") as f:
    for example in data:
        instruction = example.get("instruction", "")
        inp = example.get("input", "")
        output = example.get("output", "")
        
        messages = [
            {"role": "system", "content": "Você é um assistente útil."},
            {"role": "user", "content": f"{instruction}\n{inp}".strip()},
            {"role": "assistant", "content": output}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

print(f"Dataset criado com {len(data)} exemplos")
