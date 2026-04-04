"""
vector_store.py

Handles embeddings and vector database storage.
Uses EmbeddingsFactory to support multiple embedding providers independently.
"""

import hashlib
import os
import sys
from pathlib import Path
from langchain_community.vectorstores import Chroma

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from llm.embeddings_factory import EmbeddingsFactory
from dotenv import load_dotenv

load_dotenv()


class VectorStoreManager:

    def __init__(self):

        # Use embeddings factory instead of hardcoded provider
        embeddings = EmbeddingsFactory.get_embeddings()

        self.vectordb = Chroma(
            collection_name="enterprise_docs",
            embedding_function=embeddings,
            persist_directory="./vector_db"
        )

    def _chunk_id(self, doc):
        """Content hash + source to dedupe identical chunks across uploads."""
        source = doc.metadata.get("source", "")
        content = doc.page_content
        digest = hashlib.sha1(f"{source}|{content}".encode("utf-8", "ignore")).hexdigest()
        return digest

    def add_documents(self, documents):
        docs = [d for d in documents if getattr(d, "page_content", "") and d.page_content.strip()]
        if not docs:
            raise ValueError("No non-empty document chunks to index.")
        
        # **OPTION 3: QUOTA OPTIMIZATION**
        # Limit to 30 chunks per upload to conserve API quota
        MAX_CHUNKS_PER_UPLOAD = 30
        if len(docs) > MAX_CHUNKS_PER_UPLOAD:
            print(f"⚠️  Limiting {len(docs)} chunks to {MAX_CHUNKS_PER_UPLOAD} to preserve quota")
            docs = docs[:MAX_CHUNKS_PER_UPLOAD]
        
        deduped_docs = []
        ids = []
        for doc in docs:
            doc_id = self._chunk_id(doc)
            existing = self.vectordb._collection.get(ids=[doc_id])
            if existing and existing.get("ids"):
                continue
            deduped_docs.append(doc)
            ids.append(doc_id)
        
        if not deduped_docs:
            print("✓ All chunks already indexed")
            return
        
        print(f"📝 Indexing {len(deduped_docs)} new chunks...")
        self.vectordb.add_documents(deduped_docs, ids=ids)

    def get_retriever(self, k: int = 20):
        """
        Return a retriever. For summarizing multiple documents we prefer
        a broader fetch (higher k) with MMR search to diversify sources.
        """
        return self.vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": 50},
        )
