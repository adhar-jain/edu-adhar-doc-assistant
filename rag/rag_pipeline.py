"""rag_pipeline.py - RAG Pipeline

Combines retrieval and generation to create grounded, factual answers.

The RAG pipeline:
  1. Take user question
  2. Call retriever to find relevant chunks
  3. Construct prompt with question + retrieved context
  4. Send to LLM
  5. Parse and return grounded answer

This module ensures that answers are factually accurate based on
the documents in my knowledge base.
"""
import sys
from pathlib import Path

# Add project root to Python path for module imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag.vectordb import create_vector_store
from llm.llm_factory import LLMFactory

def build_retriever(chunks):

    vectordb = create_vector_store(chunks)

    retriever = vectordb.as_retriever(
        search_kwargs={"k": 3}
    )

    return retriever

def ask_question(retriever, question):

    llm = LLMFactory.get_llm()

    docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer the question based only on the context below.

    Context:
    {context}

    Question:
    {question}
    """

    response = llm.invoke(prompt)

    return response.content