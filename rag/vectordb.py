import sys
import os
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
    )


def create_vector_store(chunks):

    embeddings = get_embeddings()

    # ✅ HARD LIMIT to avoid Gemini quota
    chunks = chunks[:20]

    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="vector_db"
    )

    return vectordb
