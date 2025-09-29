from typing import List, Tuple
import json


class InstructionAnswerSet:
    def __init__(self, pairs: List[Tuple[str, str]]):
        self.pairs = pairs
       
    @classmethod
    def from_json(cls, json_str: str) -> 'InstructionAnswerSet':
        """Parse JSON response from LLM into domain object"""
        try:
            data = json.loads(json_str)
            pairs = [
                (pair['instruction'], pair['answer']) 
                for pair in data['instruction_answer_pairs']
            ]
            return cls(pairs)
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid JSON format for instruction-answer pairs: {e}")
    
    def __iter__(self):
        return iter(self.pairs)
    
    def __len__(self):
        return len(self.pairs)
    
    def is_valid(self) -> bool:
        """Domain rule: all pairs must have non-empty instruction and answer"""
        return all(
            instruction.strip() and answer.strip() 
            for instruction, answer in self.pairs
        )