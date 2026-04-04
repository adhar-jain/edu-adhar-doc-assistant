import sys
from pathlib import Path

# Add project root to Python path for module imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_google_genai import ChatGoogleGenerativeAI

class GeminiProvider:

    @staticmethod
    def create_llm():

        return ChatGoogleGenerativeAI(
            model="models/gemma-3-27b-it",
            temperature=0.2,
        )