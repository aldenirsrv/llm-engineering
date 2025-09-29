from dataclasses import dataclass

@dataclass
class Article:
    id: str
    content: str
    platform: str
    author_id: str
    author_full_name: str
    link: str