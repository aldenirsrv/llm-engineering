import os
import yaml
os.environ["MLX_DEVICE"] = "cpu"

import json
from pathlib import Path
from transformers import AutoTokenizer
import mlx.core as mx
from mlx_lm import load
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.tuner.datasets import load_dataset
# ==========================================
# YAML loader
# ==========================================
def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def prepare_training_data(data, output_path="training_data.jsonl"):
    """
    Prepara dados para MLX LM Tuner
    Formato: {"text": "conversação completa formatada"}
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
        
        formatted_data.append({"text": text})
    
    # Salva em JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in formatted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Dataset preparado: {len(formatted_data)} exemplos em {output_path}")
    return output_path


def train_model_mlx(
    model_name="meta-llama/Llama-3.2-3B",
    data_path="training_data.jsonl",
    output_dir="./mlx_lora_output",
    epochs=3,
    batch_size=1,
    learning_rate=1e-4,
):
    """
    Treina modelo com MLX LM Tuner
    """
    cfg = load_config("sft_config.yml")
    print("\n" + "="*60)
    print("TREINAMENTO COM MLX LM")
    print("="*60 + "\n")
    
    # Carrega modelo e tokenizer
    print(f"Carregando modelo: {model_name}")
    model, tokenizer = load(model_name)
    
    # Configuração de treinamento
    config = TrainingArgs(**{k: v for k, v in cfg.items() if k in TrainingArgs.__annotations__})
    # config = TrainingArgs(
    #     num_epochs=epochs,
    #     batch_size=batch_size,
    #     learning_rate=learning_rate,
    #     grad_checkpointing=False,
    #     save_every=100,
    #     test_every=50,
    #     seed=42,
    # )
    
    print(f"\nCarregando dataset: {data_path}")
    
    # Carrega dataset
    train_data, val_data = load_dataset(
        data_path,
        tokenizer=tokenizer,
    )
    
    print(f"Dados de treino: {len(train_data)}")
    print(f"Dados de validação: {len(val_data)}")
    
    # Cria diretório de saída
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"\nConfigurações:")
    print(f"  Épocas: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Output dir: {output_dir}\n")
    
    # Inicia treinamento
    print("Iniciando treinamento...\n")
    
    model, metrics = train(
        model=model,
        tokenizer=tokenizer,
        config=config,
        train_dataset=train_data,
        val_dataset=val_data if val_data else None,
        output_dir=output_dir,
    )
    
    print("\n" + "="*60)
    print("TREINAMENTO CONCLUÍDO")
    print("="*60)
    print(f"Modelo salvo em: {output_dir}")
    
    # Salva métricas
    metrics_path = Path(output_dir) / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Métricas salvas em: {metrics_path}\n")
    
    return model, tokenizer, metrics


def main():
    # Dados de treinamento
    training_data = [
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
        {
            "instruction": "Explique",
            "input": "O que é machine learning?",
            "output": "Machine learning é um subcampo da IA que permite aos computadores aprender com dados sem serem explicitamente programados."
        },
        {
            "instruction": "Responda",
            "input": "Quanto é 2+2?",
            "output": "2+2 é igual a 4."
        },
        {
            "instruction": "Descreva",
            "input": "Como funciona uma rede neural?",
            "output": "Uma rede neural é um modelo computacional inspirado no cérebro que aprende padrões nos dados através de camadas de neurônios artificiais."
        },
    ]
    
    # Prepara dados
    data_path = prepare_training_data(training_data, "training_data.jsonl")
    
    # Treina modelo
    model, tokenizer, metrics = train_model_mlx(
        model_name="meta-llama/Llama-3.2-3B",
        data_path=data_path,
        output_dir="./mlx_lora_output",
        epochs=2,
        batch_size=1,
        learning_rate=1e-4,
    )
    
    # Testa modelo
    print("\n" + "="*60)
    print("TESTANDO MODELO")
    print("="*60 + "\n")
    
    from mlx_lm import generate
    
    test_prompt = "Qual é a capital da Itália?"
    print(f"Prompt: {test_prompt}")
    print("\nGerando resposta...")
    
    response = generate(
        model,
        tokenizer,
        prompt=test_prompt,
        max_tokens=100,
        verbose=False
    )
    
    print(f"\nResposta:\n{response}")
    
    # Salva informações do treinamento
    info = {
        "model": "meta-llama/Llama-3.2-3B",
        "output_dir": "./mlx_lora_output",
        "epochs": 2,
        "batch_size": 1,
        "learning_rate": 1e-4,
        "training_samples": len(training_data),
        "metrics": metrics
    }
    
    with open("mlx_lora_output/training_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    print("\n✅ Treinamento finalizado com sucesso!")


if __name__ == "__main__":
    main()