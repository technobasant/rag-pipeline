"""
Embedding generation module using SentenceTransformers and Chroma vector database.
"""

import os
import json
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime

import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from pyspark.sql import SparkSession
import structlog
import ollama

from ..common.config import config
from ..common.lineage import lineage_tracker, LineageContext
from src.common.llm_client import get_embedding
from src.catalog.duckdb_catalog import create_catalog

VECTOR_DB = os.environ.get("VECTOR_DB", "chroma").lower()
EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", "auto").lower()
EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "sentence_transformers").lower()
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "nomic-embed-text")

# -------------------------  Chunking configuration  -------------------------

# Approximate token length for chunks; default 512 (roughly 350-380 words)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
# Fraction (0-1) of tokens to overlap between consecutive chunks
CHUNK_OVERLAP = float(os.getenv("CHUNK_OVERLAP", "0.2"))

# Helper to split a large text into overlapping token chunks.  Uses simple
# whitespace split for token approximation – good enough for similarity
# search with sentence-transformer embeddings.


def chunk_text(text: str, max_tokens: int = CHUNK_SIZE, overlap: float = CHUNK_OVERLAP):
    """Yield overlapping chunks from *text* capped at *max_tokens* tokens each."""
    if not text:
        return []

    tokens = text.split()
    if not tokens:
        return []

    stride = max(1, int(max_tokens * (1 - overlap)))
    chunks = []
    for start in range(0, len(tokens), stride):
        chunk_tokens = tokens[start : start + max_tokens]
        if not chunk_tokens:
            break
        chunks.append(" ".join(chunk_tokens))
        if start + max_tokens >= len(tokens):
            break
    return chunks

# Device selection logic for embeddings
def get_device():
    if EMBEDDING_DEVICE == "cpu":
        return "cpu"
    elif EMBEDDING_DEVICE == "cuda" and torch.cuda.is_available():
        return "cuda"
    else:
        return "cuda" if torch.cuda.is_available() else "cpu"

# Example vector DB switch logic
def save_embeddings(embeddings, ids):
    if VECTOR_DB == "faiss":
        try:
            import faiss
            print("[INFO] Using FAISS for vector DB.")
            dim = embeddings.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(embeddings.astype(np.float32))
            # Save index to disk (stub)
            faiss.write_index(index, "faiss.index")
        except ImportError:
            print("[ERROR] faiss not installed. Please install faiss-cpu for FAISS support.")
    else:
        try:
            print("[INFO] Using Chroma for vector DB.")
            # Existing Chroma logic here
        except ImportError:
            print("[ERROR] chromadb not installed. Please install chromadb for Chroma support.")

class EmbeddingGenerator:
    """Generate embeddings using SentenceTransformers or Ollama."""
    
    def __init__(self, model=None):
        self.model = model
        self.logger = structlog.get_logger(__name__)
        self.logger.info("Using llm_client.get_embedding for embeddings", model=self.model)

    def embed_texts(self, texts):
        embeddings = []
        for text in texts:
            try:
                emb = get_embedding(text, model=self.model)
                embeddings.append(emb)
            except Exception as e:
                self.logger.error(f"Embedding failed for text: {text[:30]}...", error=str(e))
                embeddings.append(None)
        return embeddings


class ChromaVectorStore:
    """Chroma vector database client."""
    
    def __init__(self):
        self.config = config
        self.logger = structlog.get_logger(__name__)
        self.client = None
        self.collection = None
        
    def connect(self) -> None:
        """Connect to Chroma database."""
        if self.client is None:
            self.logger.info("Connecting to Chroma", 
                           host=self.config.chroma.host,
                           port=self.config.chroma.port)
            
            self.client = chromadb.HttpClient(
                host=self.config.chroma.host,
                port=self.config.chroma.port,
                settings=Settings(allow_reset=True)
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.config.chroma.collection_name
                )
                self.logger.info("Connected to existing collection", 
                               collection=self.config.chroma.collection_name)
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.config.chroma.collection_name,
                    metadata={"description": "Document chunks for RAG"}
                )
                self.logger.info("Created new collection", 
                               collection=self.config.chroma.collection_name)
    
    def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> None:
        """Add documents with embeddings to the collection."""
        self.connect()
        
        self.logger.info("Adding documents to Chroma", 
                        document_count=len(ids))
        
        # Add documents in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_documents = documents[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            
            self.collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas
            )
            
            self.logger.debug("Added batch to Chroma", 
                            batch_start=i,
                            batch_size=len(batch_ids))
        
        self.logger.info("Successfully added all documents to Chroma")
    
    def query_similar(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Query for similar documents."""
        self.connect()
        
        # Log the incoming query request (size of embedding only to avoid dumping full vector)
        self.logger.info("Chroma similarity search", 
                        embedding_dim=len(query_embedding), 
                        n_results=n_results, 
                        where_filter=bool(where))

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )

        # Summarise the response for observability
        returned = len(results.get("ids", [[]])[0])
        top_distance = None
        if returned and results.get("distances") and results["distances"][0]:
            try:
                distance_vals = []
                for d in results["distances"][0]:
                    if d is not None:
                        try:
                            # Handle different types that ChromaDB might return
                            if isinstance(d, str):
                                distance_vals.append(float(d))
                            elif hasattr(d, '__float__'):
                                distance_vals.append(float(d))
                            else:
                                distance_vals.append(float(str(d)))
                        except (TypeError, ValueError) as item_err:
                            self.logger.warning("Unable to convert individual distance value",
                                              value=str(d),
                                              type=type(d).__name__,
                                              error=str(item_err))
                            continue
                if distance_vals:
                    top_distance = min(distance_vals)
            except (TypeError, ValueError) as conv_err:
                self.logger.warning("Unable to parse distance values from Chroma", error=str(conv_err))

        self.logger.info("Chroma similarity search complete", 
                        returned=returned, 
                        top_distance=top_distance)

        # Optional verbose debug of first few ids and scores
        import logging
        debug_enabled = logging.getLogger(__name__).isEnabledFor(logging.DEBUG)
        if debug_enabled and returned:
            preview = []
            for idx, doc_id in enumerate(results.get("ids", [[]])[0][:5]):
                distance = results.get("distances", [[]])[0][idx] if results.get("distances") else None
                preview.append({"id": doc_id, "distance": float(distance) if distance is not None else None})
            self.logger.debug("Chroma top results", preview=preview)

        return results
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        self.connect()
        
        count = self.collection.count()
        return {
            "name": self.config.chroma.collection_name,
            "count": count,
            "metadata": self.collection.metadata
        }
    
    def reset_collection(self) -> None:
        """Reset the collection (delete all documents)."""
        self.connect()
        
        self.logger.warning("Resetting collection", 
                          collection=self.config.chroma.collection_name)
        
        self.client.delete_collection(self.config.chroma.collection_name)
        self.collection = self.client.create_collection(
            name=self.config.chroma.collection_name,
            metadata={"description": "Document chunks for RAG"}
        )


class EmbeddingPipeline:
    """Complete embedding pipeline from Gold data to Chroma."""
    
    def __init__(self):
        self.config = config
        self.logger = structlog.get_logger(__name__)
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = ChromaVectorStore()
        
    def load_gold_data(self) -> pd.DataFrame:
        """Load Gold layer data from DuckDB catalog."""
        # Persist DuckDB catalog to disk
        catalog_db_path = "data/catalog.duckdb"
        catalog = create_catalog(database_path=catalog_db_path)
        gold_path = f"s3a://{self.config.minio.bucket_name}/gold/data"
        # Register the table if not already registered
        if not any(t['name'] == 'gold_table' for t in catalog.list_tables('gold')):
            catalog.register_delta_table("gold_table", gold_path, schema="gold")
        df = catalog.query("SELECT * FROM gold.gold_table")
        self.logger.info("Loaded Gold data via DuckDB for embedding", records=len(df), columns=list(df.columns))
        return df
    
    def process_embeddings(self, reset_collection: bool = False) -> Dict[str, Any]:
        """Process Gold data and generate embeddings."""
        with LineageContext(
            job_name="embedding_generation",
            tracker=lineage_tracker,
            source_code_location="src/embeddings/embedder.py"
        ) as lineage_ctx:
            
            self.logger.info("Starting embedding pipeline", reset_collection=reset_collection)
            
            # Reset collection if requested
            if reset_collection:
                self.vector_store.reset_collection()
            
            # Load Gold data
            gold_df = self.load_gold_data()
            # Deduplicate by URL to ensure unique IDs for ChromaDB
            gold_df = gold_df.drop_duplicates(subset=['url'])
            
            if gold_df.empty:
                self.logger.warning("No Gold data found for embedding generation")
                return {"status": "no_data", "embeddings_generated": 0}
            
            # ----------------------
            # New: chunk each page
            # ----------------------

            all_chunk_ids: List[str] = []
            all_texts: List[str] = []
            all_metadatas: List[Dict[str, Any]] = []

            for idx, row in gold_df.iterrows():
                page_url = str(row["url"])
                page_title = str(row["title"])
                full_text = str(row["content"])

                chunks = chunk_text(full_text)
                if not chunks:
                    continue

                total_chunks = len(chunks)
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{page_url}#chunk{i}"
                    all_chunk_ids.append(chunk_id)
                    all_texts.append(chunk)

                    all_metadatas.append({
                        "url": page_url,
                        "title": page_title,
                        "scraped_at": str(row['scraped_at']),
                        "source": str(row['source']),
                        "content_length": len(chunk),
                        "chunk_index": i,
                        "total_chunks": total_chunks,
                    })

            self.logger.info("Prepared chunks for embedding", total_chunks=len(all_texts))

            # Generate embeddings in batches to avoid memory spikes
            embeddings: List[List[float]] = []
            batch_size = 64
            for b_start in range(0, len(all_texts), batch_size):
                batch_texts = all_texts[b_start : b_start + batch_size]
                embeddings.extend(self.embedding_generator.embed_texts(batch_texts))

            if not embeddings:
                self.logger.error("No embeddings generated after chunking.")
                return None

            # Add to Chroma
            try:
                self.vector_store.add_documents(
                    ids=all_chunk_ids,
                    embeddings=embeddings,
                    documents=all_texts,
                    metadatas=all_metadatas
                )
            except Exception as e:
                # Handle duplicate ID errors and other ChromaDB exceptions
                error_msg = str(e).lower()
                if "duplicate" in error_msg or "already exists" in error_msg:
                    self.logger.warning(f"Duplicate ID error: {e}. Skipping already embedded records and continuing.")
                else:
                    self.logger.error(f"Unexpected error during ChromaDB insert: {e}")
                    # Re-raise non-duplicate errors
                    raise
            
            # Get collection info
            collection_info = self.vector_store.get_collection_info()
            
            # Add output to lineage
            output_dataset = lineage_tracker.create_output_dataset(
                name="vector_embeddings",
                namespace="chroma://documents",
                stats={
                    "embeddings_generated": len(embeddings),
                    "embedding_dimension": len(embeddings[0]) if isinstance(embeddings[0], list) else 0,
                    "collection_total_count": collection_info["count"]
                }
            )
            lineage_ctx.add_output(output_dataset)
            
            result = {
                "status": "success",
                "embeddings_generated": len(embeddings),
                "embedding_dimension": len(embeddings[0]) if isinstance(embeddings[0], list) else 0,
                "collection_info": collection_info
            }
            
            self.logger.info("Embedding pipeline completed", **result)
            return result
    
    def query_embeddings(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """Query embeddings for similar documents."""
        self.logger.info("Querying embeddings", query=query_text[:100], n_results=n_results)
        
        # Generate query embedding
        query_embedding = self.embedding_generator.embed_texts([query_text])[0]
        
        # Query Chroma
        results = self.vector_store.query_similar(
            query_embedding=query_embedding,
            n_results=n_results
        )
        
        return results
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        pass

    # ------------------------------------------------------------------
    # Back-compat helpers for legacy tests (generate_embeddings / store)
    # ------------------------------------------------------------------

    def generate_embeddings(self, texts):  # noqa: D401 – simple wrapper
        """Legacy alias – use embedding_generator.embed_texts internally."""
        return self.embedding_generator.embed_texts(texts)

    def store_embeddings(self, documents, embeddings):  # noqa: D401
        """Legacy alias – persist provided embeddings to vector store.

        This simplified implementation uses sequential IDs and empty
        metadata so older unit-tests that only assert the call succeeds
        continue to pass.  Production code should rely on process_embeddings.
        """
        if len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings length mismatch")

        ids = [f"legacy_{i}" for i in range(len(documents))]
        metadatas = [{} for _ in range(len(documents))]
        self.vector_store.add_documents(ids, embeddings, documents, metadatas)
        return True


def main():
    """Main entry point for running the embedding pipeline."""
    from ..common.config import setup_logging
    
    setup_logging(config.log_level, config.log_format)
    
    pipeline = EmbeddingPipeline()
    
    try:
        # Process embeddings
        result = pipeline.process_embeddings(reset_collection=True)
        print(f"Embedding Pipeline Results: {json.dumps(result, indent=2)}")
        
        # Test query
        if result["status"] == "success":
            query_result = pipeline.query_embeddings("What is Python?", n_results=3)
            print(f"Query Results: {json.dumps(query_result, indent=2, default=str)}")
    
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()

def embed_texts(texts, model=None):
    """
    Generate embeddings for a list of texts using LiteLLM client.
    """
    import logging
    logger = logging.getLogger(__name__)
    embeddings = []
    for text in texts:
        try:
            emb = get_embedding(text, model=model)
            embeddings.append(emb)
        except Exception as e:
            logger.error(f"Embedding failed for text: {text[:30]}... Error: {e}")
            embeddings.append(None)
    return embeddings