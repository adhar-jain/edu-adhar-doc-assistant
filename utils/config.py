"""config.py - Configuration Management

Centralized configuration for the system.

Keeps secrets secure and configuration organized.
"""
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IS_STREAMLIT_CLOUD = Path("/mount/src").is_dir()
RUNTIME_DATA_DIR = (
    Path(tempfile.gettempdir()) / "adhar-doc-assistant"
    if IS_STREAMLIT_CLOUD
    else PROJECT_ROOT
)
VECTOR_DB_DIR = RUNTIME_DATA_DIR / "vector_db"
UPLOADS_DIR = RUNTIME_DATA_DIR / "data" / "uploads"
LOG_DIR = RUNTIME_DATA_DIR / "logs"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "huggingface")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

print("LLM Provider:", LLM_PROVIDER)
print("Embeddings Provider:", EMBEDDINGS_PROVIDER)