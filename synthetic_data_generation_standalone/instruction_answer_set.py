from typing import List, Tuple
import json
class InstructionAnswerSet:
        def __init__(self, pairs:List[Tuple[str,str]]):
                self.pairs = pairs
       
        @classmethod
        def from_json(cls, json_str: str) -> 'InstructionAnswerSet':
                data = json.loads(json_str)
                pairs = [(pair['instruction'], pair['answer']) for pair in data['instruction_answer_pairs']]
                return cls(pairs)
        def __iter__(self):
                return iter(self.pairs)