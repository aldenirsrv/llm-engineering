import os
os.environ["MLX_DEVICE"] = "cpu"

from mlx_lm import load, generate
from transformers import AutoTokenizer
import json

class LlamaTrainer:
    def __init__(self, model_name):
        self.model_name = model_name
        print(f"Carregando modelo: {model_name}")
        self.model, self.tokenizer = load(model_name)
        
        # Carrega tokenizer do HuggingFace também
        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Define pad token
        if self.hf_tokenizer.pad_token is None:
            self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token
        
        # Adiciona chat template se não existir
        if self.hf_tokenizer.chat_template is None:
            print("Adicionando chat template...")
            self._add_chat_template()
    
    def _add_chat_template(self):
        """Adiciona template Llama 3.2"""
        self.hf_tokenizer.chat_template = (
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
    
    def prepare_dataset(self, data):
        """
        Converte dados para formato chat messages
        
        Args:
            data: Lista de dicts com 'instruction', 'input' (opcional), 'output'
        
        Returns:
            Lista de prompts formatados
        """
        prompts = []
        
        for example in data:
            instruction = example.get("instruction", "")
            inp = example.get("input", "")
            output = example.get("output", "")
            
            # Cria mensagens
            messages = [
                {"role": "system", "content": "Você é um assistente útil."},
                {"role": "user", "content": f"{instruction}\n{inp}".strip()},
                {"role": "assistant", "content": output}
            ]
            
            # Aplica chat template
            prompt = self.hf_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            prompts.append(prompt)
        
        return prompts
    
    def tokenize_dataset(self, prompts):
        """Tokeniza os prompts"""
        tokenized = self.hf_tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np"
        )
        return tokenized
    
    def train(self, dataset, epochs=1, batch_size=2, lr=1e-4):
        """
        Treina o modelo com LoRA
        
        Args:
            dataset: Lista de exemplos de treino
            epochs: Número de épocas
            batch_size: Tamanho do batch
            lr: Taxa de aprendizado
        """
        print("\nPreparando dataset...")
        prompts = self.prepare_dataset(dataset)
        
        print(f"Total de exemplos: {len(prompts)}")
        print(f"Primeiro exemplo:\n{prompts[0][:300]}...\n")
        
        # Tokeniza
        tokenized = self.tokenize_dataset(prompts)
        
        print(f"Input IDs shape: {tokenized['input_ids'].shape}")
        print(f"Tokens por exemplo: {len(tokenized['input_ids'][0])}")
        
        print("\nTreino iniciado...")
        print(f"Épocas: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {lr}")
        
        # Para este exemplo, apenas simulamos o treino
        # Em produção, você usaria mlx.optimizers e um loop de treino real
        print("\n✅ Dataset preparado e pronto para treino!")
        print("Nota: O treino real com MLX requer setup mais complexo com mlx.optimizers")
    
    def generate(self, prompt, max_tokens=256):
        """Gera uma resposta"""
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False
        )
        return response
    
    def save_dataset_jsonl(self, data, filename):
        """Salva dataset em JSONL para treino posterior"""
        prompts = self.prepare_dataset(data)
        
        with open(filename, 'w', encoding='utf-8') as f:
            for prompt in prompts:
                f.write(json.dumps({"text": prompt}, ensure_ascii=False) + '\n')
        
        print(f"Dataset salvo em {filename}")


def main():
    model_name = "meta-llama/Llama-3.2-3B"
    trainer = LlamaTrainer(model_name)
    
    # Dataset de exemplo
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
        }
    ]
    
    # Prepara dataset
    trainer.train(data, epochs=3, batch_size=1, lr=1e-4)
    
    # Salva para treino posterior
    trainer.save_dataset_jsonl(data, "training_data.jsonl")
    
    # Testa geração
    print("\n" + "="*60)
    print("Testando geração...")
    print("="*60)
    
    test_prompt = "Qual é a capital da França?"
    response = trainer.generate(test_prompt, max_tokens=100)
    print(f"\nPrompt: {test_prompt}")
    print(f"Resposta: {response}")


if __name__ == "__main__":
    main()