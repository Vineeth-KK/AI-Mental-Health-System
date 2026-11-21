# services/rag.py
"""Retrieval-Augmented Generation (RAG) service using ChromaDB and Sentence-Transformers."""

import os
import glob
import logging
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "knowledge_base")
CHROMA_DIR = os.path.join(BASE_DIR, "instance", "chroma_db")

class RAGService:
    def __init__(self, chroma_path: str = CHROMA_DIR):
        self.chroma_path = chroma_path
        self.collection_name = "mental_health_knowledge"
        self.client = None
        self._collection = None
        self._initialized = False
        # Lazy initialization: do not load heavy ChromaDB/embedding models into RAM at startup

    def _init_collection(self):
        if self._initialized:
            return
        # If running on Render or constrained cloud memory (512MB limit), avoid heavy ONNX model to prevent OOM
        if os.environ.get("RENDER"):
            logger.info("Render environment detected: using lightweight in-memory knowledge retrieval.")
            self._initialized = True
            self._collection = None
            return

        try:
            os.makedirs(self.chroma_path, exist_ok=True)
            if self.client is None:
                self.client = chromadb.PersistentClient(path=self.chroma_path)
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Evidence-based mental health educational resources"}
            )
            # If collection is empty, index documents
            if self._collection.count() == 0:
                self._do_index_knowledge_base()
            self._initialized = True
        except Exception as e:
            logger.warning(f"Error initializing ChromaDB collection: {e}. Fallback enabled.")
            self._initialized = False

    def _chunk_text(self, text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks by paragraph or sentences."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        for p in paragraphs:
            if len(p) <= chunk_size:
                chunks.append(p)
            else:
                words = p.split()
                current_chunk = []
                current_len = 0
                for w in words:
                    current_chunk.append(w)
                    current_len += len(w) + 1
                    if current_len >= chunk_size:
                        chunks.append(" ".join(current_chunk))
                        # overlap
                        current_chunk = current_chunk[-max(1, overlap // 10):]
                        current_len = sum(len(x) + 1 for x in current_chunk)
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
        return chunks

    def _do_index_knowledge_base(self) -> int:
        """Internal worker to index documents into an active collection."""
        if not self._collection:
            return 0
        doc_files = glob.glob(os.path.join(KNOWLEDGE_BASE_DIR, "*.txt"))
        all_ids = []
        all_texts = []
        all_metadatas = []

        chunk_counter = 0
        for file_path in doc_files:
            file_name = os.path.basename(file_path)
            topic = os.path.splitext(file_name)[0].replace("_", " ").title()
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                chunks = self._chunk_text(content)
                for idx, chunk in enumerate(chunks):
                    chunk_id = f"{file_name}_{idx}"
                    all_ids.append(chunk_id)
                    all_texts.append(chunk)
                    all_metadatas.append({
                        "source": file_name,
                        "topic": topic,
                        "chunk_index": idx
                    })
                    chunk_counter += 1
            except Exception as e:
                logger.error(f"Failed to read/index {file_path}: {e}")

        if all_texts:
            self._collection.upsert(
                ids=all_ids,
                documents=all_texts,
                metadatas=all_metadatas
            )
            logger.info(f"Successfully indexed {len(all_texts)} chunks into ChromaDB.")
        return len(all_texts)

    def index_knowledge_base(self) -> int:
        """Read knowledge_base/*.txt files and index them into ChromaDB."""
        if not self._initialized:
            self._init_collection()
        return self._do_index_knowledge_base()

    def query(self, query_text: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Query ChromaDB for relevant educational resources."""
        if not self._initialized:
            self._init_collection()

        if not self._initialized or not self._collection or self._collection.count() == 0:
            # Fallback keyword matching across knowledge_base directory
            return self._fallback_query(query_text, n_results)

        try:
            results = self._collection.query(
                query_texts=[query_text],
                n_results=min(n_results, self._collection.count())
            )
            
            output = []
            if results and "documents" in results and results["documents"]:
                docs = results["documents"][0]
                metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
                dists = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(docs)
                
                for doc, meta, dist in zip(docs, metas, dists):
                    # ChromaDB default L2 distance typically ranges 0 to 2 for normalized embeddings
                    rel = round(max(0.0, min(1.0, 1.0 - (float(dist) / 2.0))), 3) if dist is not None else 0.85
                    output.append({
                        "text": doc,
                        "topic": meta.get("topic", "General Wellbeing"),
                        "source": meta.get("source", "Evidence-Based Guidance"),
                        "relevance_score": rel
                    })
            return output if output else self._fallback_query(query_text, n_results)
        except Exception as e:
            logger.warning(f"ChromaDB query failed: {e}. Using fallback query.")
            return self._fallback_query(query_text, n_results)

    def _fallback_query(self, query_text: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Reliable pure-python fallback for keyword relevance."""
        doc_files = glob.glob(os.path.join(KNOWLEDGE_BASE_DIR, "*.txt"))
        scored = []
        tokens = set(query_text.lower().split())

        for file_path in doc_files:
            file_name = os.path.basename(file_path)
            topic = os.path.splitext(file_name)[0].replace("_", " ").title()
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                chunks = self._chunk_text(content)
                for chunk in chunks:
                    chunk_tokens = set(chunk.lower().split())
                    common = len(tokens.intersection(chunk_tokens))
                    if common > 0:
                        scored.append((common, chunk, topic, file_name))
            except Exception:
                continue

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, chunk, topic, file_name in scored[:n_results]:
            results.append({
                "text": chunk,
                "topic": topic,
                "source": file_name,
                "relevance_score": 0.80
            })
        
        # If no keywords matched, return general stress & anxiety advice
        if not results:
            results.append({
                "text": "Try the 4-7-8 breathing exercise: Inhale quietly for 4 seconds, hold gently for 7 seconds, exhale completely for 8 seconds. This resets your autonomic nervous system.",
                "topic": "Anxiety Management",
                "source": "anxiety_management.txt",
                "relevance_score": 0.75
            })
        return results

# Singleton instance
rag_service = RAGService()
