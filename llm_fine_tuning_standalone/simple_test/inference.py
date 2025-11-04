from mlx_lm import load, generate

# # Load fine-tuned model with LoRA adapters
model, tokenizer = load(
    "meta-llama/Llama-3.2-3B-Instruct",
    adapter_path="./adapters"
)
# 🔹 base model
# model, tokenizer = load("meta-llama/Llama-3.2-3B-Instruct")
test_prompts = [
    # "User: Outline the process of deploying a new model in a continuous training environment.\nAssistant:",
    # "User: What is the significance of the Bytewax streaming engine in the feature pipeline?\nAssistant:",
    # "User: Explain the role of feature stores in fine tuning LLMs.\nAssistant:",
    # "User: Discuss the significance of data preparation in building machine learning models.\nAssistant:",
    "User: Explain the importance of a vector database in RAG.\nAssistant:",
    # "User:  Explain what does RAG acronym means.\nAssistant:"
    

]

for prompt in test_prompts:
    result = generate(model, tokenizer, prompt, max_tokens=200)
    print("🔹 Prompt:", prompt)
    print("🔸 Model Output:", result)  # ✅ no ["text"], just print directly
    print("-" * 80)