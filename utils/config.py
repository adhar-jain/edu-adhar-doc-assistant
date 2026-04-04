"""config.py - Configuration Management

Centralized configuration for the system.

Keeps secrets secure and configuration organized.
"""
import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "huggingface")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

print("LLM Provider:", LLM_PROVIDER)
print("Embeddings Provider:", EMBEDDINGS_PROVIDER)