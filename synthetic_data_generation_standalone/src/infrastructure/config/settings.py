from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL_ID", "gpt-4o-mini")
    hunggingface_access_token: str = os.getenv("HUGGINGFACE_ACCESS_TOKEN", "")