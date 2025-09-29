from datasets import Dataset
from ..infrastructure.config.settings import Settings
from .di_container import DIContainer

def main(dataset_id: str) -> Dataset:
    settings = Settings()
    container = DIContainer(settings)
    
    return container.create_instruction_dataset_use_case.execute(
        "data/articles.json",  # Updated path
        dataset_id
    )

if __name__ == "__main__":
    result = main("aldenirsrv/llmtwin")
    print("Instruction dataset:")
    print("Train split:")
    print(result['train'].to_pandas())
    print(f"Train samples: {len(result['train'])}")
    print("\nTest split:")
    print(result['test'].to_pandas())
    print(f"Test samples: {len(result['test'])}")