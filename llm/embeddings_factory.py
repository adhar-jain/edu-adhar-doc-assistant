"""
embeddings_factory.py

Factory for creating embeddings providers without affecting LLM factory.
Supports: gemini, openai, huggingface
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import EMBEDDINGS_PROVIDER as DEFAULT_EMBEDDINGS_PROVIDER


class EmbeddingsFactory:

    @staticmethod
    def get_embeddings():
        """
        Get embeddings based on the configured provider.
        Uses environment variable if set, otherwise config default.
        """

        # Check environment variable first (for dynamic selection)
        provider = os.getenv("EMBEDDINGS_PROVIDER", DEFAULT_EMBEDDINGS_PROVIDER)

        if provider == "gemini":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                api_key=os.getenv("GOOGLE_API_KEY")
            )

        elif provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=os.getenv("OPENAI_API_KEY")
            )

        elif provider == "huggingface":
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model="all-MiniLM-L6-v2",
                # Runs locally - no API key needed
            )

        else:
            raise ValueError(f"Unsupported embeddings provider: {provider}")
