# 📘 Fine-Tuning Module (Clean Architecture)

This repository provides a **Clean Architecture implementation** for fine-tuning and inference with large language models (LLMs) using [Unsloth](https://github.com/unslothai/unsloth), [Hugging Face Transformers](https://huggingface.co/docs/transformers/index), and [TRL](https://huggingface.co/docs/trl/index).  

The goal is to separate **domain logic**, **use cases**, and **infrastructure dependencies**, ensuring scalability, testability, and maintainability.

---

## 📂 Project Structure

```
src/
├── domain/
│   ├── entities/              # Core business objects (Model, Dataset)
│   └── services/              # Business rules (training, inference logic)
├── application/
│   ├── interfaces/            # Abstract repositories/interfaces
│   └── use_cases/             # Orchestrators (finetune, inference, save model)
├── infrastructure/
│   ├── repositories/          # Hugging Face / dataset implementations
│   ├── external/              # Adapters (Unsloth, Torch, etc.)
│   └── config/                # Environment & settings
└── main/
    └── cli.py                 # Entry point (CLI interface)
```

This modular design enforces **dependency inversion**:
- **Domain** knows nothing about infrastructure.  
- **Application** orchestrates workflows through **interfaces**.  
- **Infrastructure** implements those interfaces with concrete dependencies (Hugging Face, Torch, etc.).  

---

## 🚀 Features

- ✅ **Supervised Fine-Tuning (SFT)** and **Direct Preference Optimization (DPO)**  
- ✅ **LoRA adapters** for efficient training  
- ✅ **Dummy mode** for quick testing (400 samples only)  
- ✅ **Hugging Face Hub integration** (push/pull models)  
- ✅ **Inference pipeline** with chat templates  
- ✅ **Clean separation** of domain, application, and infrastructure  

---

## ⚙️ Installation

1. Clone the repo:
   ```bash
   https://github.com/aldenirsrv/llm-engineering/tree/llm_fine_tuning/llm_fine_tuning_standalone.git
   cd llm_fine_tuning
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Configure environment variables:
   ```bash
   export SM_MODEL_DIR=./models
   export SM_OUTPUT_DATA_DIR=./outputs
   export SM_NUM_GPUS=1
   ```

---

## 📦 Requirements

### Core
```
torch>=2.0.0
datasets>=2.14.0
transformers>=4.41.0
trl>=0.9.0
```

### Hugging Face Hub
```
huggingface_hub>=0.24.0
```

### Unsloth
```
unsloth>=0.7.0
```

### Typing extensions (compatibility)
```
typing-extensions>=4.8.0
```

---

## 🌍 Environment Variables

These variables configure authentication and integrations:

| Variable | Purpose |
|----------|---------|
| `HUGGINGFACE_ACCESS_TOKEN` | Required to pull/push models to [Hugging Face Hub](https://huggingface.co/) |
| `COMET_API_KEY` | Required if you want to log experiments with [Comet ML](https://www.comet.com/) |
| `SM_MODEL_DIR` | Directory to save trained models (default: `./models`) |
| `SM_OUTPUT_DATA_DIR` | Directory to store training outputs (default: `./outputs`) |
| `SM_NUM_GPUS` | Number of GPUs available for training (default: `1`) |

Set them before running:
```bash
export HUGGINGFACE_ACCESS_TOKEN=hf_xxx
export COMET_API_KEY=sk_xxx
```

# 📊 Comet Integration

This module includes optional integration with [Comet ML](https://www.comet.com/site/) for experiment tracking and model management.

## What is Comet?

Comet is a platform that helps you track, compare, and optimize your machine learning experiments. With Comet integration, you can:

* **Log training metrics** – Track loss, accuracy, and custom metrics in real time
* **Monitor hyperparameters** – Record configuration settings for each experiment
* **Store artifacts** – Save model checkpoints, visualizations, and other outputs
* **Compare runs** – Evaluate performance across multiple experiments side by side
* **Version models** – Collaborate with your team using centralized model storage

## Setup

### With Comet Tracking (Optional)

To enable experiment tracking, set your Comet API key as an environment variable:
```bash
export COMET_API_KEY=your_api_key_here

---

## 🖥️ Usage

### Run CLI

```bash
python -m src.main.cli   --finetuning_type sft   --num_train_epochs 3   --per_device_train_batch_size 2   --learning_rate 3e-4   --dataset_huggingface_workspace mlabonne   --model_output_huggingface_workspace your-hf-username   --is_dummy False
```

### Example: SFT in Dummy Mode
```bash
python -m src.main.cli --finetuning_type sft --is_dummy True
```

### Example: DPO in Dummy Mode
```bash
python -m src.main.cli --finetuning_type dpo --is_dummy True
```

---

## 📚 Architecture in Action

- **Domain** → defines `ModelEntity` and rules like EOS token handling.  
- **Application** → orchestrates **finetuning_use_case**, **inference_use_case**, etc.  
- **Infrastructure** → implements Hugging Face integration (`HuggingFaceModelRepository`).  
- **Main** → runs the pipeline via `cli.py`.  

---

## 🧪 Testing

Run unit tests:

```bash
pytest tests/
```

Each layer can be tested independently:
- Mock `ModelRepository` to test domain logic.  
- Test services (`TrainingService`, `InferenceService`) without Hugging Face dependencies.  

---

## ☁️ Hugging Face Hub Integration

You can push models automatically after training:

```python
save_model(model, tokenizer, "output_dir", push_to_hub=True, repo_id="your-hf-username/model-name")
```

Make sure you are logged in:

```bash
huggingface-cli login
```

---

## 🤝 Contributing

1. Fork the project  
2. Create a feature branch (`git checkout -b feat/my-feature`)  
3. Commit with **Conventional Commits**  
   ```
   feat(training): add support for gradient checkpointing
   fix(inference): corrected EOS token usage
   ```
4. Push and open a PR 🎉  

---

## 📜 License

MIT License © 2025 – Aldenir Flauzino / Engineering Potion   
