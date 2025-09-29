from abc import ABC, abstractmethod
from typing import List
from ...domain.entities.article import Article

class ArticleRepository(ABC):
    @abstractmethod
    def load_from_file(self, file_path: str) -> List[Article]:
        pass