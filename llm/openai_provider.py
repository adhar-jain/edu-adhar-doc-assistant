import sys
from pathlib import Path

# Add project root to Python path for module imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_openai import ChatOpenAI
from utils.config import OPENAI_API_KEY

class OpenAIProvider:

    @staticmethod
    def create_llm():

        return ChatOpenAI(
            # model="gpt-4o-mini",
            model="text-embedding-3-small",
            api_key=OPENAI_API_KEY
        )