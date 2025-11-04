# 🚀 LLM Fine-tuning with Clean Architecture

A production-ready implementation for fine-tuning Large Language Models (LLMs) using Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) with clean architecture principles.

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Comet Integration](#comet-integration)

## ✨ Features

- **Dual Fine-tuning Strategies**: Support for both SFT and DPO training
- **Clean Architecture**: Organized codebase following domain-driven design principles
- **4-bit Quantization**: Memory-efficient training with LoRA adapters
- **Experiment Tracking**: Optional Comet ML integration for monitoring
- **Hub Integration**: Seamless model publishing to Hugging Face Hub
- **Flexible Configuration**: Command-line arguments and environment variables

## 🏗️ Architecture

This project follows clean architecture principles with clear separation of concerns:

```
src/
├── domain/              # Business logic and entities
│   ├── entities/        # Core domain models
│   └── services/        # Domain services
├── application/         # Application-specific business rules
│   ├── interfaces/      # Abstract interfaces
│   └── use_cases/       # Use case implementations
├── infrastructure/      # External interfaces and implementations
│   ├── repositories/    # Data access layer
│   ├── external/        # Third-party integrations
│   └── config/          # Configuration management
└── main/                # Application entry points
```

## 📦 Requirements

### Core Dependencies

```txt
# Deep Learning Framework
torch>=2.0.0

# Dataset and Model Management
datasets>=2.14.0
transformers>=4.41.0
trl>=0.9.0

# Hugging Face Hub Integration
huggingface_hub>=0.24.0

# Efficient Fine-tuning
unsloth>=0.7.0

# Type Safety
typing-extensions>=4.8.0
```

### Python Version

- Python 3.8 or higher

## 🔧 Installation

1. **Clone the repository**

```bash
git clone <repository-url>
cd <repository-name>
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

## 🔐 Environment Setup

Create a `.env` file in the root directory with the following configuration:

```bash
# Hugging Face API Configuration
# Purpose: Authentication for downloading models and pushing fine-tuned models to the Hub
HUGGINGFACE_ACCESS_TOKEN=hf_your_token_here

# Comet ML API Configuration (Optional)
# Purpose: Experiment tracking, metrics logging, and model versioning
COMET_API_KEY=sk_your_key_here
```

### Environment Variable Purposes

| Variable | Required | Purpose |
|----------|----------|---------|
| `HUGGINGFACE_ACCESS_TOKEN` | ✅ Yes | Authenticates access to Hugging Face Hub for model downloads and uploads |
| `COMET_API_KEY` | ⚠️ Optional | Enables experiment tracking and visualization in Comet ML dashboards |

### Getting Your API Keys

**Hugging Face Token:**
1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a new token with write permissions
3. Copy and add to `.env` file

**Comet API Key:**
1. Sign up at [comet.com](https://www.comet.com/signup)
2. Navigate to Settings → API Keys
3. Copy your API key and add to `.env` file

## 📁 Project Structure

```
.
├── src/
│   ├── domain/
│   │   ├── entities/
│   │   │   ├── __init__.py
│   │   │   ├── model_config.py
│   │   │   └── training_config.py
│   │   └── services/
│   │       ├── __init__.py
│   │       └── model_service.py
│   ├── application/
│   │   ├── interfaces/
│   │   │   ├── __init__.py
│   │   │   ├── model_repository.py
│   │   │   └── dataset_repository.py
│   │   └── use_cases/
│   │       ├── __init__.py
│   │       ├── train_sft.py
│   │       └── train_dpo.py
│   ├── infrastructure/
│   │   ├── repositories/
│   │   │   ├── __init__.py
│   │   │   ├── huggingface_model_repository.py
│   │   │   └── huggingface_dataset_repository.py
│   │   ├── external/
│   │   │   ├── __init__.py
│   │   │   └── comet_tracker.py
│   │   └── config/
│   │       ├── __init__.py
│   │       └── settings.py
│   └── main/
│       ├── __init__.py
│       └── train.py
├── .env
├── requirements.txt
└── README.md
```

## 🚀 Usage

### Supervised Fine-Tuning (SFT)

Train a model using supervised fine-tuning:

```bash
python src/main/train.py \
  --finetuning_type sft \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --learning_rate 3e-4 \
  --dataset_huggingface_workspace your-workspace \
  --model_output_huggingface_workspace your-workspace
```

### Direct Preference Optimization (DPO)

Fine-tune a model using DPO (requires a pre-trained SFT model):

```bash
python src/main/train.py \
  --finetuning_type dpo \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --learning_rate 2e-6 \
  --dataset_huggingface_workspace your-workspace \
  --model_output_huggingface_workspace your-workspace
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--finetuning_type` | str | `sft` | Training strategy: `sft` or `dpo` |
| `--num_train_epochs` | int | `3` | Number of training epochs |
| `--per_device_train_batch_size` | int | `2` | Batch size per GPU |
| `--learning_rate` | float | `3e-4` | Learning rate for optimizer |
| `--dataset_huggingface_workspace` | str | `mlabonne` | HF workspace for datasets |
| `--model_output_huggingface_workspace` | str | `mlabonne` | HF workspace for model output |
| `--is_dummy` | bool | `False` | Use reduced dataset for testing |

### Testing Mode

For quick testing with a reduced dataset:

```bash
python src/main/train.py \
  --finetuning_type sft \
  --is_dummy True
```

## 📊 Comet Integration

This module includes optional integration with [Comet ML](https://www.comet.com/site/) for experiment tracking and model management.

### What is Comet?

Comet helps you track, compare, and optimize your machine learning experiments:

* **Log training metrics** – Track loss, accuracy, and custom metrics in real time
* **Monitor hyperparameters** – Record configuration settings for each experiment
* **Store artifacts** – Save model checkpoints, visualizations, and other outputs
* **Compare runs** – Evaluate performance across multiple experiments side by side
* **Version models** – Collaborate with your team using centralized model storage

### Setup

#### With Comet Tracking (Optional)

To enable experiment tracking, set your Comet API key as an environment variable:

```bash
export COMET_API_KEY=your_api_key_here
```

Once configured, your training runs will automatically be logged to your Comet dashboard.

#### Without Comet Tracking

If you don't set the `COMET_API_KEY` environment variable, training will proceed normally without any experiment tracking. The integration is completely optional and won't affect your training pipeline.

### Example Usage

```bash
# Enable Comet tracking
export COMET_API_KEY=sk_xxx

# Run your training script
python src/main/train.py --finetuning_type sft
```

Your experiment data will now be available in your Comet workspace for analysis and collaboration.

## 🤝 Contributing

Contributions are welcome! Please follow the clean architecture principles when adding new features.

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning
- [Hugging Face](https://huggingface.co) for model and dataset hosting
- [Comet ML](https://www.comet.com) for experiment tracking

---

**Note**: Ensure you have sufficient GPU memory for training. The default configuration uses 4-bit quantization with LoRA to minimize memory usage.