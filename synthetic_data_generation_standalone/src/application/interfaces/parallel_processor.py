from abc import ABC, abstractmethod
from typing import List
from ...domain.entities.text_extract import TextExtract
from ...domain.entities.instruction_answer_pair import InstructionAnswerPair
from ...domain.services.instruction_generation_service import InstructionGenerationService

class ParallelProcessor(ABC):
    @abstractmethod
    def process_extracts(
        self, 
        extracts: List[TextExtract], 
        service: InstructionGenerationService
    ) -> List[InstructionAnswerPair]:
        pass
