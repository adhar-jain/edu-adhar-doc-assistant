"""
main.py

Application entry point for the GenAI Adhar-Docs-Assistant.

Responsibilities:
- Streamlit UI
- Document upload
- Document ingestion
- Chunking
- Vector DB storage
- Agent execution
- Safety checks
"""

import sys
import os
import hashlib
import logging
import re
from logging.handlers import RotatingFileHandler
import streamlit as st
import shutil

# Ensure project root (genai-enterprise-knowledge-agent) is on PYTHONPATH
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

LOG_DIR = os.path.join(ROOT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
APP_LOG_PATH = os.path.join(LOG_DIR, "answer_errors.log")
logger = logging.getLogger("doc_assistant.answer")
logger.setLevel(logging.ERROR)
if not logger.handlers:
    handler = RotatingFileHandler(
        APP_LOG_PATH,
        maxBytes=1_000_000,
        backupCount=2,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)

from ingestion.document_loader import load_document
from ingestion.chunker import chunk_document
from vectorstore.vector_store import VectorStoreManager
from llm.llm_factory import LLMFactory
from agents.rag_agent import build_rag_agent
from utils.config import (
    EMBEDDINGS_PROVIDER as DEFAULT_EMBEDDINGS_PROVIDER,
    LLM_PROVIDER as DEFAULT_LLM_PROVIDER,
)


@st.cache_resource(show_spinner="Loading selected language model...")
def load_llm(provider):
    return LLMFactory.get_llm(provider)


@st.cache_resource(show_spinner="Loading selected embeddings model...")
def load_vector_store(provider):
    os.environ["EMBEDDINGS_PROVIDER"] = provider
    return VectorStoreManager()



# -----------------------------------
# Provider Selection Dashboard
# -----------------------------------

st.title("🤖 Adhar-Docs-Assistant")

# Language model selection
if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = DEFAULT_LLM_PROVIDER

st.subheader("💬 Choose Your Language Model")

col_hf_llm, col_gemini_llm = st.columns(2)

with col_hf_llm:
    if st.button("🤗 HuggingFace Local\nFree to run\nUses this computer", key="llm_huggingface"):
        load_llm.clear()
        st.session_state.llm_provider = "huggingface"
    if st.session_state.llm_provider == "huggingface":
        st.success("✅ Currently Selected")
        st.caption(
            "Qwen2.5-1.5B-Instruct runs locally, using CUDA when available or CPU otherwise. "
            "The first use downloads the model weights."
        )

with col_gemini_llm:
    if st.button("🤖 Google Gemini\nCloud-Based\nUses API quota", key="llm_gemini"):
        load_llm.clear()
        st.session_state.llm_provider = "gemini"
    if st.session_state.llm_provider == "gemini":
        st.success("✅ Currently Selected")

# Initialize session state for provider selection
if "embeddings_provider" not in st.session_state:
    st.session_state.embeddings_provider = DEFAULT_EMBEDDINGS_PROVIDER

# Provider selection with cards
st.subheader("🧠 Choose Your Embeddings Provider")

col1, col2 = st.columns(2)

with col1:
    if st.button("🤗 HuggingFace\nFree & Local\nNo API quota", key="hf"):
        if st.session_state.embeddings_provider != "huggingface":
            st.session_state.embeddings_provider = "huggingface"
            st.info("📁 Switched to HuggingFace. Its vector index is kept separate.")
    if st.session_state.embeddings_provider == "huggingface":
        st.success("✅ Currently Selected")

with col2:
    if st.button("🤖 Google Gemini\nCloud-Based\nRequires API key", key="gemini"):
        if st.session_state.embeddings_provider != "gemini":
            st.session_state.embeddings_provider = "gemini"
            st.info("📁 Switched to Gemini. Its vector index is kept separate.")
    if st.session_state.embeddings_provider == "gemini":
        st.success("✅ Currently Selected")

# Update environment variable
os.environ["EMBEDDINGS_PROVIDER"] = st.session_state.embeddings_provider

st.markdown("---")

# Manual clear button
col_clear, col_spacer = st.columns([1, 5])
with col_clear:
    if st.button("🗑️ Clear Vector DB", help="Clear the vector database and start fresh"):
        try:
            if os.path.exists("./vector_db"):
                # Use ignore_errors=True to force delete even if files are locked
                shutil.rmtree("./vector_db", ignore_errors=True)
                st.session_state.processed_uploads = set()
                load_vector_store.clear()
                st.success("✅ Vector database cleared! Restarting app...")
                st.rerun()
            else:
                st.info("📁 No database to clear")
        except Exception as e:
            st.error(f"Error clearing database: {e}")

st.markdown("---")

# -----------------------------------
# Initialize components
# -----------------------------------

# Initialize LLM
try:
    llm = load_llm(st.session_state.llm_provider)
except Exception as e:
    logger.exception("Language model initialization failed")
    error_message = str(e)
    for key_name in ("GOOGLE_API_KEY", "OPENAI_API_KEY"):
        secret = os.getenv(key_name)
        if secret:
            error_message = error_message.replace(secret, "[redacted]")
    error_message = re.sub(r"AIza[0-9A-Za-z_-]{20,}|sk-[A-Za-z0-9_-]{20,}", "[redacted]", error_message)
    st.error(
        f"Could not initialize {st.session_state.llm_provider}: "
        f"{type(e).__name__}: {error_message}"
    )
    with open(APP_LOG_PATH, "rb") as log_file:
        st.download_button(
            "Download initialization error log",
            data=log_file.read(),
            file_name="answer_errors.log",
            mime="text/plain",
            key="download_initialization_error_log",
            on_click="ignore",
        )
    st.stop()

# Initialize vector database
vector_store = load_vector_store(st.session_state.embeddings_provider)

# Create retriever
retriever = vector_store.get_retriever()

# Build LangGraph agent
agent = build_rag_agent(llm, retriever)

# Chat memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------------
# File Upload
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload a document for analysis. you can upload 1 or more documents. Supported formats: PDF, TXT, CSV, XLSX.",
    type=["pdf", "txt", "csv", "xlsx"]
)

if uploaded_file:
    if "processed_uploads" not in st.session_state:
        st.session_state.processed_uploads = set()

    upload_digest = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    upload_id = f"{st.session_state.embeddings_provider}:{upload_digest}"
    if upload_id not in st.session_state.processed_uploads:
        os.makedirs("data/uploads", exist_ok=True)
        file_path = f"data/uploads/{uploaded_file.name}"

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success("Document uploaded successfully!")

        try:
            document_text = load_document(file_path)
            chunks = chunk_document(document_text, uploaded_file.name)
            vector_store.add_documents(chunks)
            st.session_state.processed_uploads.add(upload_id)
            st.success("Document processed and stored in knowledge base.")
        except Exception as e:
            st.error(f"Failed to process document: {str(e)}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            st.stop()
    else:
        st.info("This document is already processed for the selected embeddings provider.")

# -----------------------------------
# Question Input
# -----------------------------------

with st.form("question_form"):
    question = st.text_input("Ask a question about your documents")
    ask_question = st.form_submit_button("Ask")

if ask_question:
    if not question.strip():
        st.warning("Please enter a valid question.")
    else:
        # Agent input state
        state = {
            "question": question,
            "chat_history": st.session_state.chat_history
        }

        try:

            result = agent.invoke(state)

            answer = result["answer"]

            st.write("### Answer")
            st.write(answer)
            sources = result.get("sources", [])

            if sources:

                st.write("### Sources")

                for s in sources:
                    st.write(f"- {s}")
            # Update memory
            st.session_state.chat_history.append(
                {"question": question, "answer": answer}
            )

        except Exception:
            logger.exception("Answer generation failed")
            st.error("An error occurred while generating the answer.")
            with open(APP_LOG_PATH, "rb") as log_file:
                st.download_button(
                    "Download answer error log",
                    data=log_file.read(),
                    file_name="answer_errors.log",
                    mime="text/plain",
                    key="download_answer_error_log",
                    on_click="ignore",
                )

# -----------------------------------
# Summarize each document
# -----------------------------------

if st.button("Summarize each document"):
    try:
        with st.status("Loading indexed document chunks...", expanded=True) as status:
            raw = vector_store.vectordb._collection.get(include=["documents", "metadatas"])
            docs = raw.get("documents", []) or []
            metas = raw.get("metadatas", []) or []
            grouped = {}
            for content, meta in zip(docs, metas):
                source = meta.get("source", "unknown")
                grouped.setdefault(source, []).append(content)
            status.update(
                label=f"Loaded {len(docs)} chunks from {len(grouped)} documents",
                state="complete",
                expanded=False,
            )

        if not grouped:
            st.warning("No documents indexed yet. Please upload files first.")
        else:
            st.write("### Summaries by document")
            total_documents = len(grouped)
            progress = st.progress(0, text=f"Starting summaries for {total_documents} documents...")
            for index, (source, chunks) in enumerate(grouped.items(), start=1):
                # Keep prompt reasonable
                joined = "\n\n".join(chunks)
                prompt = f"""You are summarizing a single document.
Document source: {source}

Content:
{joined}

Provide a concise summary with 4-6 bullet points."""
                progress.progress(
                    (index - 1) / total_documents,
                    text=f"Summarizing document {index} of {total_documents}: {source}",
                )
                st.write(f"**{source}**")
                st.write_stream(llm.stream(prompt))
                st.write("---")
                progress.progress(
                    index / total_documents,
                    text=f"Finished document {index} of {total_documents}",
                )
    except Exception as e:
        st.error(f"Failed to summarize documents: {e}")

# -----------------------------------
# Chat History
# -----------------------------------

if st.session_state.chat_history:

    st.write("### Conversation History")

    for chat in reversed(st.session_state.chat_history):

        st.write("**User:**", chat["question"])
        st.write("**Assistant:**", chat["answer"])
        st.write("---")
