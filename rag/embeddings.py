import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from utils.config import GOOGLE_API_KEY


def get_embedding_model():

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        GOOGLE_API_KEY=GOOGLE_API_KEY
    )

    return embeddings
