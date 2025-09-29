from dataclasses import dataclass

@dataclass
class TextExtract:
    content: str
    min_length: int = 1000
    max_length: int = 2000