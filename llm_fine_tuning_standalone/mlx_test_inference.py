# import os
# os.environ["MLX_DEVICE"] = "cpu"
# from mlx_lm import load, generate
# from transformers import AutoTokenizer

# def main():
#     # Troque para a versão Instruct
#     model_name = "meta-llama/Llama-3.2-3B-Instruct"  # ← Adicione -Instruct
    
#     print(f"Carregando modelo: {model_name}")
#     model, tokenizer = load(model_name)
#     hf_tokenizer = AutoTokenizer.from_pretrained(model_name)

#     messages = [
#         {"role": "system", "content": "Você é um assistente útil e preciso."},
#         {"role": "user", "content": "Explique a teoria da relatividade em termos simples."}
#     ]
    
#     # Aplica o chat template (a versão Instruct já tem pronto)
#     prompt = hf_tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )
    
#     print("Gerando resposta...\n")
    
#     response = generate(
#         model, 
#         tokenizer, 
#         prompt=prompt, 
#         verbose=False,
#         max_tokens=1000
#     )
    
#     print("Resposta:\n", response)

# if __name__ == "__main__":
#     main()


import os
os.environ["MLX_DEVICE"] = "cpu"
from mlx_lm import load, generate

def main():
    model_name = "meta-llama/Llama-3.2-3B"
    
    print(f"Carregando modelo: {model_name}")
    model, tokenizer = load(model_name)
    
    # ❌ NÃO faça assim (chat):
    # prompt = "<|start_header_id|>user<|end_header_id|>\nExplique a relatividade<|eot_id|>"
    
    # ✅ FAÇA assim (completação de texto):
    prompt = "Explique a teoria da relatividade em termos simples:\n"
    
    print("Gerando resposta...\n")
    
    response = generate(
        model, 
        tokenizer, 
        prompt=prompt,
        verbose=True,
        max_tokens=256
    )
    
    print("Resposta:\n", response)

if __name__ == "__main__":
    main()