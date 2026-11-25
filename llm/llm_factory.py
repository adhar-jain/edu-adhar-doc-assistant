import sys
import os
from pathlib import Path

# Add project root to Python path for module imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import LLM_PROVIDER as DEFAULT_LLM_PROVIDER
from llm.huggingface_provider import HuggingFaceProvider
from llm.openai_provider import OpenAIProvider
from llm.gemini_provider import GeminiProvider


class LLMFactory:

    @staticmethod
    def get_llm(provider=None):
        """
        Get LLM based on the requested or configured provider.
        """

        provider = provider or os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER)

        if provider == "openai":
            return OpenAIProvider.create_llm()

        elif provider == "gemini":
            return GeminiProvider.create_llm()

        elif provider == "huggingface":
            return HuggingFaceProvider.create_llm()

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")