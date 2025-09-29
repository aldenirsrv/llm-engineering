from dataclasses import dataclass

@dataclass
class InstructionAnswerPair:
    instruction: str
    answer: str