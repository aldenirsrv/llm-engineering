from abc import ABC, abstractmethod
from typing import List
from datasets import Dataset
from ...domain.entities.instruction_answer_pair import InstructionAnswerPair

class DatasetRepository(ABC):
    @abstractmethod
    def create_from_pairs(self, pairs: List[InstructionAnswerPair]) -> Dataset:
        pass
    
    @abstractmethod
    def train_test_split(self, dataset: Dataset, test_size: float) -> Dataset:
        pass
    
    @abstractmethod
    def save_to_hub(self, dataset: Dataset, dataset_id: str) -> None:
        pass