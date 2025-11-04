# LLM Fine-tuning with Clean Architecture

A professional, maintainable implementation of LLM fine-tuning using SFT (Supervised Fine-Tuning) and DPO (Direct Preference Optimization) with clean architecture principles.

## 🏗️ Architecture

This project implements **Clean Architecture** with clear separation of concerns:

- **Configuration Layer**: Type-safe configuration objects
- **Service Layer**: Business logic (Model, Dataset, Training services)
- **Repository Layer**: External API interactions (HuggingFace Hub)
- **Entry Point**: CLI interface and workflow orchestration

## 🚀 Features

- ✅ **SFT Training**: Supervised fine-tuning with Alpaca-style prompts
- ✅ **DPO Training**: Direct Preference Optimization
- ✅ **Clean Architecture**: SOLID principles, testable, maintainable
- ✅ **Type Safety**: Full type hints and validation
- ✅ **Configurable**: Easy to customize via configuration objects
- ✅ **Production Ready**: Proper error handling, logging, and monitoring
- ✅ **Efficient**: Uses Unsloth for optimized training
- ✅ **LoRA Support**: Parameter-efficient fine-tuning

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-finetuning.git
cd llm-finetuning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install

# For development
make install-dev
```

## 🎯 Quick Start

### SFT Training
```bash
python main.py \
    --finetuning_type sft \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --learning_rate 3e-4 \
    --dataset_huggingface_workspace your-workspace \
    --model_output_huggingface_workspace your-workspace
```

### DPO Training
```bash
python main.py \
    --finetuning_type dpo \
    --dataset_huggingface_workspace your-workspace \
    --model_output_huggingface_workspace your-workspace
```

### Dummy Mode (Testing)
```bash
python main.py --finetuning_type sft --is_dummy True
```

Or use Makefile shortcuts:
```bash
make run-sft
make run-dpo
make run-sft-dummy
make run-dpo-dummy
```

## 📝 Configuration

### Model Configuration
```python
from src.config import ModelConfig

config = ModelConfig(
    model_name="meta-llama/Llama-3.1-8B",
    max_seq_length=2048,
    load_in_4bit=False,
    lora_rank=32,
    lora_alpha=32,
    lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
    chat_template="chatml",
)
```

### Training Configuration
```python
from src.config import TrainingConfig

config = TrainingConfig(
    finetuning_type="sft",
    output_dir="./output",
    dataset_workspace="your-workspace",
    learning_rate=3e-4,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    is_dummy=False,
)
```

## 🧪 Testing

```bash
# Run all tests
make test

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_model_service.py -v
```

### Example Test
```python
# tests/test_model_service.py
import pytest
from unittest.mock import Mock, MagicMock, patch
from src.services.model_service import ModelService
from src.config import ModelConfig
from src.repositories.huggingface_repository import HuggingFaceRepository


class TestModelService:
    @pytest.fixture
    def mock_hf_repo(self):
        return Mock(spec=HuggingFaceRepository)
    
    @pytest.fixture
    def model_service(self, mock_hf_repo):
        return ModelService(mock_hf_repo)
    
    @pytest.fixture
    def model_config(self):
        return ModelConfig(
            model_name="test-model",
            max_seq_length=512,
            lora_rank=8,
        )
    
    @patch('src.services.model_service.FastLanguageModel')
    def test_load_model(self, mock_fast_lm, model_service, model_config):
        # Arrange
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_fast_lm.from_pretrained.return_value = (mock_model, mock_tokenizer)
        mock_fast_lm.get_peft_model.return_value = mock_model
        
        # Act
        model, tokenizer = model_service.load_model(model_config)
        
        # Assert
        assert model is not None
        assert tokenizer is not None
        mock_fast_lm.from_pretrained.assert_called_once()
        mock_fast_lm.get_peft_model.assert_called_once()
    
    def test_save_model_local_only(self, model_service):
        # Arrange
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        output_dir = "test_output"
        
        # Act
        model_service.save_model(
            mock_model, 
            mock_tokenizer, 
            output_dir, 
            push_to_hub=False
        )
        
        # Assert
        mock_model.save_pretrained_merged.assert_called_once_with(
            output_dir, 
            mock_tokenizer, 
            save_method="merged_16bit"
        )
        mock_model.push_to_hub_merged.assert_not_called()
```

## 🔧 Development

### Code Quality
```bash
# Format code
make format

# Run linting
make lint

# Type checking
mypy src/

# All quality checks
make format && make lint && make test
```

### Project Structure
```
project_root/
├── main.py                          # Entry point
├── requirements.txt                 # Dependencies
├── setup.py                        # Package setup
├── Makefile                        # Common commands
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # Configuration models
│   │
│   ├── services/                   # Business logic
│   │   ├── model_service.py
│   │   ├── dataset_service.py
│   │   └── training_service.py
│   │
│   └── repositories/               # Data access
│       └── huggingface_repository.py
│
└── tests/                          # Test suite
    ├── test_model_service.py
    ├── test_dataset_service.py
    ├── test_training_service.py
    └── test_huggingface_repository.py
```

## 🎓 Key Concepts

### Clean Architecture Benefits
1. **Testability**: Each component can be tested in isolation
2. **Maintainability**: Changes are localized
3. **Extensibility**: Easy to add new features
4. **Reusability**: Services can be used independently
5. **Type Safety**: Catch errors early with type hints

### SOLID Principles Applied
- **Single Responsibility**: Each class has one job
- **Open/Closed**: Open for extension, closed for modification
- **Liskov Substitution**: Services can be swapped
- **Interface Segregation**: Focused interfaces
- **Dependency Inversion**: Depend on abstractions

## 📊 Monitoring

Training metrics are reported to Comet ML. Configure your API key:

```bash
export COMET_API_KEY=your_api_key
export COMET_PROJECT_NAME=your_project
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting (`make test && make lint`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Unsloth](https://github.com/unslothai/unsloth) for efficient training
- Uses [Transformers](https://github.com/huggingface/transformers) and [TRL](https://github.com/huggingface/trl)
- Inspired by clean architecture principles from Uncle Bob

## 📚 Resources

- [Clean Architecture by Robert C. Martin](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [SOLID Principles](https://en.wikipedia.org/wiki/SOLID)
- [Unsloth Documentation](https://docs.unsloth.ai/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Enable `load_in_4bit=True`
- Reduce `max_seq_length`

### Model Not Found
- Check your HuggingFace token is configured
- Verify the model name is correct
- Ensure you have access to the model

### Import Errors
- Reinstall dependencies: `make install`
- Check Python version (>= 3.9)
- Verify virtual environment is activated

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues and discussions
- Consult the documentation

---

**Happy Fine-tuning! 🚀**
