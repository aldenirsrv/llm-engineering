# Project Setup Guide

This project follows **Clean Architecture principles**, ensuring a clear separation of concerns and proper dependency direction (inner layers never depend on outer layers).

---

## 📂 Directory Structure

Run the following commands to create the directory layout:

```bash
# Source code
mkdir -p src/{domain/{entities,services},application/{interfaces,use_cases},infrastructure/{repositories,external,config},main}

# Tests
mkdir -p tests/{unit/{domain,application,infrastructure},integration}

# Data
mkdir -p data
```

## 🛠️ Optional: Automate Setup with a Shell Script

To simplify project setup, you can create a shell script to run the directory creation commands automatically.

### 1. Create the script file
Create a file named **`clean_arch.sh`** in the project root and add the following content:

```bash
#!/bin/bash
# Clean Architecture Project Setup Script

# Source code structure
mkdir -p src/{domain/{entities,services},application/{interfaces,use_cases},infrastructure/{repositories,external,config},main}

# Tests
mkdir -p tests/{unit/{domain,application,infrastructure},integration}

# Data
mkdir -p data

echo "✅ Project structure created successfully!"
```

### 2. Make the script executable
In the terminal, run:
```bash
chmod +x clean_arch.sh
```

### 3. Run the script
You can execute the script in two ways:
```bash
# Run as standalone script
./clean_arch.sh

# OR run within the current shell session
source clean_arch.sh
```
#### Final structure overview:

```bash
src/
 ├── domain/
 │   ├── entities/
 │   └── services/
 ├── application/
 │   ├── interfaces/
 │   └── use_cases/
 ├── infrastructure/
 │   ├── repositories/
 │   ├── external/
 │   └── config/
 └── main/

tests/
 ├── unit/
 │   ├── domain/
 │   ├── application/
 │   └── infrastructure/
 └── integration/

data/
```

---

## 📦 Move Existing Files

- Move `articles.json` → `data/`  
- Keep `config.py`, `utils.py`, and `instruction_answer_set.py` in the **project root**  
  (or adapt them into `src/infrastructure/config/` if preferred).  

---

## 🔗 Update Imports

Ensure your Python imports use **relative paths** according to the new structure.  
Example:  

```python
from src.domain.entities import SomeEntity
from src.application.use_cases import SomeUseCase
from src.infrastructure.repositories import SomeRepository
```

---

## ▶️ Run the Project

From the project root, run:

```bash
python -m src.main.main
```

This ensures Python resolves imports correctly with the package structure.

---

## 🧪 Testing

This project uses **pytest** for testing.  

### Install dependencies:
```bash
pip install pytest
```

### Run all tests:
```bash
pytest
```

### Run only unit tests:
```bash
pytest tests/unit
```

### Run only integration tests:
```bash
pytest tests/integration
```

---

✅ Following this setup, you’ll have a **modular, maintainable, and scalable** foundation for development.

## 📂 Directory Structure
Directory Structure
```bash
synthetic_data_generation_standalone/
├── src/
│   ├── __init__.py
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── entities/
│   │   │   ├── __init__.py
│   │   │   ├── article.py
│   │   │   ├── text_extract.py
│   │   │   └── instruction_answer_pair.py
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── text_extraction_service.py
│   │       └── instruction_generation_service.py
│   ├── application/
│   │   ├── __init__.py
│   │   ├── interfaces/
│   │   │   ├── __init__.py
│   │   │   ├── article_repository.py
│   │   │   ├── dataset_repository.py
│   │   │   ├── llm_client.py
│   │   │   └── parallel_processor.py
│   │   └── use_cases/
│   │       ├── __init__.py
│   │       └── create_instruction_dataset.py
│   ├── infrastructure/
│   │   ├── __init__.py
│   │   ├── repositories/
│   │   │   ├── __init__.py
│   │   │   ├── json_article_repository.py
│   │   │   └── huggingface_dataset_repository.py
│   │   ├── external/
│   │   │   ├── __init__.py
│   │   │   ├── openai_client.py
│   │   │   └── thread_pool_processor.py
│   │   └── config/
│   │       ├── __init__.py
│   │       └── settings.py
│   └── main/
│       ├── __init__.py
│       ├── di_container.py
│       └── main.py
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── domain/
│   │   ├── application/
│   │   └── infrastructure/
│   └── integration/
├── data/
│   └── articles.json
├── requirements.txt
├── pyproject.toml
└── README.md
```