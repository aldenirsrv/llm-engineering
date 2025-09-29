import re
from typing import List
from ..entities.article import Article
from ..entities.text_extract import TextExtract
from ...infrastructure.utils import clean_text  # Keep your existing utils
        

class TextExtractionService:
    def __init__(self, min_length: int = 1000, max_length: int = 2000):
        self.min_length = min_length
        self.max_length = max_length
    
    def extract_chunks(self, articles: List[Article]) -> List[TextExtract]:
       
        extracts = []
        sentence_pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s"

        for article in articles:
            cleaned_article = clean_text(article.content)
            sentences = re.split(sentence_pattern, cleaned_article)
            current_chunk = ""
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                if len(current_chunk) + len(sentence) <= self.max_length:
                    current_chunk += sentence + " "
                else:
                    if len(current_chunk) >= self.min_length:
                        extracts.append(TextExtract(current_chunk.strip()))
                    current_chunk = sentence + " "
                    
            if len(current_chunk) >= self.min_length:
                extracts.append(TextExtract(current_chunk.strip()))
                
        return extracts