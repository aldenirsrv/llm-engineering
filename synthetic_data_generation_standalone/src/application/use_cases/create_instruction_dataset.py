from typing import List
from datasets import Dataset
from ..interfaces.article_repository import ArticleRepository
from ..interfaces.dataset_repository import DatasetRepository
from ..interfaces.parallel_processor import ParallelProcessor
from ...domain.services.text_extraction_service import TextExtractionService
from ...domain.services.instruction_generation_service import InstructionGenerationService

class CreateInstructionDatasetUseCase:
    def __init__(
        self,
        article_repository: ArticleRepository,
        dataset_repository: DatasetRepository,
        text_extraction_service: TextExtractionService,
        instruction_generation_service: InstructionGenerationService,
        parallel_processor: ParallelProcessor
    ):
        self.article_repository = article_repository
        self.dataset_repository = dataset_repository
        self.text_extraction_service = text_extraction_service
        self.instruction_generation_service = instruction_generation_service
        self.parallel_processor = parallel_processor

    def execute(self, input_file_path: str, dataset_id: str) -> Dataset:
        # Load articles
        articles = self.article_repository.load_from_file(input_file_path)
        
        # Extract text chunks
        extracts = self.text_extraction_service.extract_chunks(articles)
        
        # Generate instruction-answer pairs in parallel
        instruction_pairs = self.parallel_processor.process_extracts(
            extracts, 
            self.instruction_generation_service
        )
        
        if not instruction_pairs:
            raise RuntimeError("No instruction–answer pairs were generated.")
        
        # Create and save dataset
        dataset = self.dataset_repository.create_from_pairs(instruction_pairs)
        split_dataset = self.dataset_repository.train_test_split(dataset, test_size=0.1)
        self.dataset_repository.save_to_hub(split_dataset, dataset_id)
        
        return split_dataset