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
import streamlit as st
import shutil

# Ensure project root (genai-enterprise-knowledge-agent) is on PYTHONPATH
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from ingestion.document_loader import load_document
from ingestion.chunker import chunk_document
from vectorstore.vector_store import VectorStoreManager
from llm.llm_factory import LLMFactory
from agents.rag_agent import build_rag_agent
from utils.config import EMBEDDINGS_PROVIDER as DEFAULT_EMBEDDINGS_PROVIDER



# -----------------------------------
# Provider Selection Dashboard
# -----------------------------------

st.title("🤖 Adhar-Docs-Assistant")

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
            # Clear vector DB when switching
            if os.path.exists("./vector_db"):
                shutil.rmtree("./vector_db")
                st.success("🗑️ Switched to HuggingFace - Vector DB cleared")
            else:
                st.info("📁 Switched to HuggingFace")
    if st.session_state.embeddings_provider == "huggingface":
        st.success("✅ Currently Selected")

with col2:
    if st.button("🤖 Google Gemini\nCloud-Based\nRequires API key", key="gemini"):
        if st.session_state.embeddings_provider != "gemini":
            st.session_state.embeddings_provider = "gemini"
            # Clear vector DB when switching
            if os.path.exists("./vector_db"):
                shutil.rmtree("./vector_db")
                st.success("🗑️ Switched to Gemini - Vector DB cleared")
            else:
                st.info("📁 Switched to Gemini")
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
llm = LLMFactory.get_llm()

# Initialize vector database
vector_store = VectorStoreManager()

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

    os.makedirs("data/uploads", exist_ok=True)

    file_path = f"data/uploads/{uploaded_file.name}"

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("Document uploaded successfully!")

    try:

        # Load document
        document_text = load_document(file_path)

        # Chunk document
        chunks = chunk_document(
                        document_text,
                        uploaded_file.name
                    )

        # Store in vector DB
        vector_store.add_documents(chunks)

        st.success("Document processed and stored in knowledge base.")

    except Exception as e:

        st.error(f"Failed to process document: {str(e)}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        st.stop()

# -----------------------------------
# Question Input
# -----------------------------------

question = st.text_input("Ask a question about your documents")

if question:

    # Input validation
    if not question.strip():

        st.warning("Please enter a valid question.")
        st.stop()

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

    except Exception as e:

        st.error("An error occurred while generating the answer.")

# -----------------------------------
# Summarize each document
# -----------------------------------

if st.button("Summarize each document"):
    try:
        raw = vector_store.vectordb._collection.get(include=["documents", "metadatas"])
        docs = raw.get("documents", []) or []
        metas = raw.get("metadatas", []) or []
        # Group chunks by source
        grouped = {}
        for content, meta in zip(docs, metas):
            source = meta.get("source", "unknown")
            grouped.setdefault(source, []).append(content)

        if not grouped:
            st.warning("No documents indexed yet. Please upload files first.")
        else:
            st.write("### Summaries by document")
            for source, chunks in grouped.items():
                # Keep prompt reasonable
                joined = "\n\n".join(chunks)
                prompt = f"""You are summarizing a single document.
Document source: {source}

Content:
{joined}

Provide a concise summary with 4-6 bullet points."""
                resp = llm.invoke(prompt)
                st.write(f"**{source}**")
                st.write(resp.content)
                st.write("---")
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
