from openai import OpenAI
from ...application.interfaces.llm_client import LLMClient

class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def generate(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant who generates instruction-answer pairs based on the given context. Provide your response in JSON format."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=1200,
            temperature=0.7,
        )
        return completion.choices[0].message.content