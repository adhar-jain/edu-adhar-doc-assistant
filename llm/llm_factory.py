import sys
import os
from pathlib import Path

# Add project root to Python path for module imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import LLM_PROVIDER as DEFAULT_LLM_PROVIDER


class LLMFactory:

    @staticmethod
    def get_llm(provider=None):
        """
        Get LLM based on the requested or configured provider.
        """

        provider = provider or os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER)

        if provider == "openai":
            from llm.openai_provider import OpenAIProvider
            return OpenAIProvider.create_llm()

        elif provider == "gemini":
            from llm.gemini_provider import GeminiProvider
            return GeminiProvider.create_llm()

        elif provider == "huggingface":
            from llm.huggingface_provider import HuggingFaceProvider
            return HuggingFaceProvider.create_llm()

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")