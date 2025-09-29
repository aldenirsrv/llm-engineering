from ..infrastructure.repositories.json_article_repository import JsonArticleRepository
from ..infrastructure.repositories.huggingface_dataset_repository import HuggingFaceDatasetRepository
from ..infrastructure.external.openai_client import OpenAIClient
from ..infrastructure.external.thread_pool_processor import ThreadPoolParallelProcessor
from ..domain.services.text_extraction_service import TextExtractionService
from ..domain.services.instruction_generation_service import InstructionGenerationService
from ..application.use_cases.create_instruction_dataset import CreateInstructionDatasetUseCase

class DIContainer:
    def __init__(self, config):
        self.config = config
        self._setup_dependencies()
    
    def _setup_dependencies(self):
        # Infrastructure
        self.llm_client = OpenAIClient(
            api_key=self.config.openai_api_key,
            model=self.config.openai_model or "gpt-4o-mini"
        )
        self.article_repository = JsonArticleRepository()
        self.dataset_repository = HuggingFaceDatasetRepository()
        self.parallel_processor = ThreadPoolParallelProcessor(max_workers=4)
        
        # Domain Services
        self.text_extraction_service = TextExtractionService()
        self.instruction_generation_service = InstructionGenerationService(self.llm_client)
        
        # Use Cases
        self.create_instruction_dataset_use_case = CreateInstructionDatasetUseCase(
            self.article_repository,
            self.dataset_repository,
            self.text_extraction_service,
            self.instruction_generation_service,
            self.parallel_processor
        )
