import concurrent.futures
from typing import List
from tqdm.auto import tqdm
from ...application.interfaces.parallel_processor import ParallelProcessor
from ...domain.entities.text_extract import TextExtract
from ...domain.entities.instruction_answer_pair import InstructionAnswerPair
from ...domain.services.instruction_generation_service import InstructionGenerationService

class ThreadPoolParallelProcessor(ParallelProcessor):
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
    
    def process_extracts(
        self, 
        extracts: List[TextExtract], 
        service: InstructionGenerationService
    ) -> List[InstructionAnswerPair]:
        instruction_answer_pairs = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(service.generate_pairs, extract) 
                for extract in extracts
            ]
            
            with tqdm(total=len(futures), desc="Generating instruction/answer pairs") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            instruction_answer_pairs.extend(result)
                    except Exception as e:
                        print(f"Error in future: {e}")
                    finally:
                        pbar.update(1)
        
        return instruction_answer_pairs