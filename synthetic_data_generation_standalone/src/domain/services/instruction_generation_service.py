from typing import List
from ..entities.text_extract import TextExtract
from ..entities.instruction_answer_pair import InstructionAnswerPair
from ..entities.instruction_answer_set import InstructionAnswerSet
from ...application.interfaces.llm_client import LLMClient


class InstructionGenerationService:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def _build_prompt(self, extract_content: str) -> str:
        return f"""Based on the following extract, generate five instruction-answer pairs. Each instruction \
            must ask to write about a specific topic contained in the context. each answer \
            must provide a relevant paragraph based on the information found in the \
            context. Only use concepts from the context to generate the instructions. \
            Instructions must never explicitly mention a context, a system, a course, or an extract. \
            Instructions must be self-contained and general. \
            Answers must imitate the writing style of the context. \
            Example instruction: Explain the concept of an LLM Twin.
            Example answer: An LLM Twin is essentially an AI character that mimics your writing style, personality, and voice. \
            It's designed to write just like you by incorporating these elements into a language model. \
            The idea is to create a digital replica of your writing habits using advanced AI techniques. \
            Provide your response in JSON format with the following structure:
            {{
            "instruction_answer_pairs": [
            {{"instruction": "...", "answer": "..."}},
            ...
            ]
            }}
            Extract:
            {extract_content}
        """
    
    def _parse_response(self, response: str) -> List[InstructionAnswerPair]:
        result = InstructionAnswerSet.from_json(response)
        return [InstructionAnswerPair(pair[0], pair[1]) for pair in result.pairs]

    def generate_pairs(self, extract: TextExtract) -> List[InstructionAnswerPair]:
        prompt = self._build_prompt(extract.content)
        response = self.llm_client.generate(prompt)
        return self._parse_response(response)