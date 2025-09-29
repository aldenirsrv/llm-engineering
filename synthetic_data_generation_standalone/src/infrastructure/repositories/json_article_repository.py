import json
from typing import List
from ...application.interfaces.article_repository import ArticleRepository
from ...domain.entities.article import Article

class JsonArticleRepository(ArticleRepository):
    def load_from_file(self, file_path: str) -> List[Article]:
        with open(file_path, "r") as file:
            data = json.load(file)
        
        return [
            Article(
                id=item["id"],
                content=item["content"],
                platform=item["platform"],
                author_id=item["author_id"],
                author_full_name=item["author_full_name"],
                link=item["link"]
            )
            for item in data["artifact_data"]
        ]