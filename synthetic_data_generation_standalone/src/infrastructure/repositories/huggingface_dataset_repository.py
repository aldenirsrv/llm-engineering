from typing import List
from datasets import Dataset
from ...application.interfaces.dataset_repository import DatasetRepository
from ...domain.entities.instruction_answer_pair import InstructionAnswerPair

class HuggingFaceDatasetRepository(DatasetRepository):
    def create_from_pairs(self, pairs: List[InstructionAnswerPair]) -> Dataset:
        instructions = [pair.instruction for pair in pairs]
        answers = [pair.answer for pair in pairs]
        
        return Dataset.from_dict({
            "instruction": instructions,
            "output": answers
        })
    
    def train_test_split(self, dataset: Dataset, test_size: float) -> Dataset:
        return dataset.train_test_split(test_size=test_size)
    
    def save_to_hub(self, dataset: Dataset, dataset_id: str) -> None:
        dataset.push_to_hub(dataset_id)