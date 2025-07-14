"""
FastAPI RAG service with Ollama LLM integration.
"""

import json
import asyncio
import time
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import httpx
import structlog
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import os

# Handle imports for both direct execution and module import
try:
    from ..common.config import config, setup_logging
    from .metrics import (
        track_request_metrics, track_rag_query, track_vector_search,
        track_llm_request, get_active_connections, get_metrics_registry
    )
    from ..embeddings.embedder import EmbeddingPipeline
    from ..common.llm_client import get_embedding, get_completion, chat, stream_chat
except ImportError:
    # Direct execution fallback
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from common.config import config, setup_logging
    from api.metrics import (
        track_request_metrics, track_rag_query, track_vector_search,
        track_llm_request, get_active_connections, get_metrics_registry
    )
    from embeddings.embedder import EmbeddingPipeline
    from common.llm_client import get_embedding, get_completion, chat, stream_chat


# Setup logging
setup_logging(config.log_level, config.log_format)
logger = structlog.get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Data-Scrape-to-RAG API",
    description="RAG API for querying scraped documentation with local LLM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for chat UI
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logger.info("Mounted static files - directory: %s", static_dir)

# Global embedding pipeline instance
embedding_pipeline: Optional[EmbeddingPipeline] = None

# Default similarity threshold for relevance filtering
SIMILARITY_DEFAULT = 0.25


# --------------------------
# Helper utilities
# --------------------------

# Convert cosine *distance* returned by Chroma (range 0 .. 2) into a
# 0‒1 similarity score where 1.0 means identical and 0.0 maximally different.
# Formula: similarity = 1 - (distance / 2).  Values outside expected range are
# clamped.  This centralises the logic so we avoid duplicating `1.0 - d` and
# can tweak behaviour in one place.

def distance_to_similarity(distance_val: float) -> float:
    """Map Chroma cosine distance → similarity in range [0, 1]."""
    try:
        dv = float(distance_val)
    except (TypeError, ValueError):
        return 0.0

    # Cosine distance theoretically lies within 0..2
    if dv <= 0:
        return 1.0
    if dv >= 2:
        return 0.0
    return 1.0 - (dv / 2.0)


class RAGResponse(BaseModel):
    """Response model for RAG queries."""
    answer: str = Field(..., description="Generated answer from the LLM")
    sources: List[Dict[str, Any]] = Field(..., description="Source documents used for the answer")
    query: str = Field(..., description="Original query")
    response_time: float = Field(..., description="Response time in seconds")
    timestamp: str = Field(..., description="Response timestamp")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Health check timestamp")
    services: Dict[str, str] = Field(..., description="Status of dependent services")
    version: str = Field(..., description="API version")


class OllamaClient:
    """Client for interacting with Ollama LLM."""
    
    def __init__(self):
        self.base_url = f"http://{config.ollama.host}:{config.ollama.port}"
        self.model = config.ollama.model
        self.logger = structlog.get_logger(__name__)
    
    async def generate_response(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> str:
        """Generate response from Ollama LLM."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                        "top_p": 0.9,
                        "stop": ["Human:", "Assistant:", "\n\nHuman:", "\n\nAssistant:"]
                    }
                }
                
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                response.raise_for_status()
                result = response.json()
                
                return result.get("response", "").strip()
                
        except httpx.RequestError as e:
            self.logger.error("Ollama request failed: %s", str(e), exc_info=True)
            raise HTTPException(status_code=503, detail=f"LLM service unavailable: {str(e)}")
        except httpx.HTTPStatusError as e:
            self.logger.error(
                "Ollama HTTP error (status %s): %s",
                e.response.status_code,
                str(e),
                exc_info=True,
            )
            raise HTTPException(status_code=502, detail=f"LLM service error: {str(e)}")
    
    async def generate_response_stream(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response from Ollama LLM."""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                        "top_p": 0.9,
                        "stop": ["Human:", "Assistant:", "\n\nHuman:", "\n\nAssistant:"]
                    }
                }
                
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/generate",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                if data.get("response"):
                                    yield data["response"]
                                if data.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                continue
                                
        except httpx.RequestError as e:
            self.logger.error("Ollama streaming request failed: %s", str(e), exc_info=True)
            yield f"\n\n[Error: LLM service unavailable - {str(e)}]"
        except httpx.HTTPStatusError as e:
            self.logger.error(
                "Ollama streaming HTTP error (status %s): %s",
                e.response.status_code,
                str(e),
                exc_info=True,
            )
            yield f"\n\n[Error: LLM service error - {str(e)}]"
    
    async def health_check(self) -> bool:
        """Check if Ollama service is healthy."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False


# Global Ollama client
ollama_client = OllamaClient()


def get_embedding_pipeline() -> EmbeddingPipeline:
    """Dependency to get embedding pipeline instance."""
    global embedding_pipeline
    if embedding_pipeline is None:
        embedding_pipeline = EmbeddingPipeline()
    return embedding_pipeline


def create_rag_prompt(query: str, context_docs: List[Dict[str, Any]]) -> str:
    """Create RAG prompt with context and query."""
    context_text = ""
    
    for i, doc in enumerate(context_docs, 1):
        metadata = doc.get('metadata', {})
        document_text = doc.get('document', '')
        
        context_text += f"\n--- Source {i} ---\n"
        context_text += f"Title: {metadata.get('title', 'Unknown')}\n"
        context_text += f"URL: {metadata.get('url', 'Unknown')}\n"
        context_text += f"Content: {document_text}\n"
    
    prompt = f"""You are a helpful assistant that answers questions based on the provided context. Use only the information from the context to answer the question. If the context doesn't contain enough information to answer the question, say so.

Context:
{context_text}

Question: {query}

Answer: Based on the provided context, """
    
    return prompt


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting RAG API service")
    
    # Initialize embedding pipeline
    global embedding_pipeline
    embedding_pipeline = EmbeddingPipeline()
    
    logger.info("RAG API service started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down RAG API service")
    
    if embedding_pipeline:
        embedding_pipeline.cleanup()
    
    logger.info("RAG API service shutdown complete")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with timeouts to prevent hanging."""
    services = {}
    
    # Check Ollama with timeout
    try:
        ollama_healthy = await asyncio.wait_for(
            ollama_client.health_check(),
            timeout=3.0
        )
        services["ollama"] = "healthy" if ollama_healthy else "unhealthy"
    except asyncio.TimeoutError:
        services["ollama"] = "unhealthy: timeout"
    except Exception as e:
        services["ollama"] = f"unhealthy: {str(e)}"
    
    # Check Chroma with timeout (run in thread pool to avoid blocking)
    try:
        def check_chroma():
            try:
                pipeline = get_embedding_pipeline()
                # First try to connect to ChromaDB
                pipeline.vector_store.connect()
                
                # Try to get collection info
                collection_info = pipeline.vector_store.get_collection_info()
                return {"status": "healthy", "info": collection_info}
            except Exception as e:
                # Fallback: direct HTTP heartbeat request to determine liveness even if SDK fails
                try:
                    import requests
                    heartbeat_url = f"http://{config.chroma.host}:{config.chroma.port}/api/v1/heartbeat"
                    resp = requests.get(heartbeat_url, timeout=2)
                    # Treat 200 OK as healthy; 404 Not Found still means the server is reachable but the route is missing
                    if resp.status_code in (200, 404):
                        # Server reachable, now attempt to list collections via HTTP client for richer info
                        try:
                            import chromadb
                            from chromadb.config import Settings
                            client = chromadb.HttpClient(
                                host=config.chroma.host,
                                port=config.chroma.port,
                                settings=Settings(allow_reset=True)
                            )
                            collections = client.list_collections()
                            collection_names = [col.name for col in collections]
                            if config.chroma.collection_name in collection_names:
                                return {"status": "connected_collection_exists", "collections": collection_names}
                            return {"status": "connected_no_target_collection", "collections": collection_names, "target": config.chroma.collection_name}
                        except Exception:
                            # Could not list collections but server is responding
                            return {"status": "reachable_no_collections"}
                    else:
                        return {"status": "connection_failed", "error": f"HTTP {resp.status_code}"}
                except Exception as conn_e:
                    return {"status": "connection_failed", "error": str(conn_e)}
        
        # Run the synchronous ChromaDB call in a thread pool with timeout
        chroma_result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, check_chroma),
            timeout=5.0
        )
        
        if chroma_result["status"] == "healthy":
            services["chroma"] = "healthy"
            services["chroma_documents"] = str(chroma_result["info"].get("count", 0))
        elif chroma_result["status"] == "connected_collection_exists":
            services["chroma"] = "connected (collection exists)"
            services["chroma_collections"] = str(len(chroma_result.get("collections", [])))
        elif chroma_result["status"] == "connected_no_target_collection":
            services["chroma"] = f"connected (target collection '{chroma_result.get('target', 'unknown')}' not found)"
            services["chroma_collections"] = str(len(chroma_result.get("collections", [])))
        elif chroma_result["status"] == "connected_no_collections":
            services["chroma"] = "connected (no collections exist yet)"
        elif chroma_result["status"] == "reachable_no_collections":
            services["chroma"] = "reachable (no collections available)"
        else:
            services["chroma"] = f"unhealthy: {chroma_result.get('error', 'unknown error')}"
            
    except asyncio.TimeoutError:
        services["chroma"] = "unhealthy: timeout"
    except Exception as e:
        services["chroma"] = f"unhealthy: {str(e)}"
    
    overall_status = "healthy" if all(
        status == "healthy" or status.startswith("healthy")
        for status in services.values()
        if not status.isdigit()
    ) else "unhealthy"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat(),
        services=services,
        version="1.0.0"
    )


@app.get("/ask")
def ask(
    question: str,
    top_k: int = 3,
    similarity_threshold: float = Query(SIMILARITY_DEFAULT, ge=0.0, le=1.0, description="Minimum similarity threshold for relevant documents"),
    include_meta: bool = Query(False, description="Include additional metadata fields in response")
):
    """
    Enhanced RAG endpoint with fallback logic and similarity thresholds.
    Provides intelligent responses even when no relevant documents are found.
    """
    import logging
    logger = logging.getLogger(__name__)
    start_time = time.time()
    
    try:
        # Ensure similarity_threshold is properly converted to float
        threshold_val = float(similarity_threshold)
        
        # 1. Embed the question using llm_client
        question_embedding = get_embedding(question)
        
        # 2. Retrieve top_k sources from ChromaDB
        vector_start = time.time()
        pipeline = EmbeddingPipeline()
        results = pipeline.vector_store.query_similar(
            query_embedding=question_embedding,
            n_results=top_k
        )
        track_vector_search(time.time() - vector_start)
        
        # 3. Filter results by similarity threshold
        relevant_docs = []
        relevant_sources = []
        
        if results.get('documents') and results['documents'][0]:
            documents = results['documents'][0]
            metadatas = results['metadatas'][0] if results.get('metadatas') else [{}] * len(documents)
            distances = results['distances'][0] if results.get('distances') else [1.0] * len(documents)
            
            for doc, metadata, distance in zip(documents, metadatas, distances):
                # Chroma may return distances as Decimal or str; coerce to float
                try:
                    distance_val = float(distance)
                except (TypeError, ValueError):
                    # If conversion fails skip this entry with warning
                    logger.warning("Invalid distance value from Chroma - value: %s, type: %s", str(distance), type(distance).__name__)
                    continue
                
                # Convert Chroma distance → similarity (0..1)
                similarity_score = distance_to_similarity(distance_val)
                
                if similarity_score >= threshold_val:
                    relevant_docs.append(doc)
                    content_preview = doc[:200] + "..." if len(doc) > 200 else doc
                    relevant_sources.append({
                        # Promote common metadata fields to top-level keys the UI expects
                        'title': metadata.get('title', f'Document {len(relevant_sources)+1}'),
                        'url': metadata.get('url', ''),
                        'content': doc,
                        'content_preview': content_preview,
                        'similarity_score': float(similarity_score),
                        'metadata': metadata
                    })
                    
                logger.debug("Document similarity",
                           similarity_score=similarity_score,
                           threshold=threshold_val,
                           relevant=similarity_score >= threshold_val)
        
        # 4. Generate answer with context or fallback
        llm_start = time.time()
        
        if relevant_docs:
            # Use RAG with relevant context
            context = "\n\n".join(relevant_docs)
            prompt = f"""Based on the following context, please answer the question. If the context doesn't contain enough information to fully answer the question, please say so and provide what information you can.

Context: {context}

Question: {question}

Answer:"""
            logger.info("Using RAG with context - question: %s, relevant_docs_count: %d",
                       question[:50], len(relevant_docs))
        else:
            # Fallback to general knowledge
            prompt = f"""I don't have specific documents in my knowledge base that are relevant to your question (similarity threshold: {threshold_val}). However, I can provide a general answer based on my training:

Question: {question}

Please provide a helpful answer based on general knowledge, and mention that this response is not based on specific documents in the knowledge base."""
            logger.info("Using fallback response - question: %s, threshold: %f",
                       question[:50], threshold_val)
        
        answer = get_completion(prompt)
        track_llm_request("llama3.2:1b", True, time.time() - llm_start, len(answer.split()))
        
        # Track overall RAG query
        track_rag_query(True, time.time() - start_time, len(relevant_docs))
        
        # Core response (required by spec)
        response = {
            "answer": answer,
            "sources": relevant_sources,
            "query": question,
            "response_time": time.time() - start_time,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Optional diagnostic fields for UI/debugging
        if include_meta:
            response.update({
                "context_used": len(relevant_docs) > 0,
                "similarity_threshold": threshold_val,
                "total_documents_found": len(results.get('documents', [[]])[0]),
                "relevant_documents_count": len(relevant_docs)
            })
        
        return response
        
    except Exception as e:
        # Avoid passing unsupported keyword arguments to the stdlib logger
        logger.error(
            "RAG API error while answering question '%s': %s",
            question[:50],
            str(e),
            exc_info=True,
        )
        track_rag_query(False, time.time() - start_time, 0)
        
        # Provide fallback response even on error
        try:
            fallback_prompt = f"""There was an error accessing the knowledge base. Please provide a helpful general answer to this question:

Question: {question}

Note: This response is based on general knowledge only, not specific documents."""
            
            fallback_answer = get_completion(fallback_prompt)
            return {
                "answer": fallback_answer,
                "sources": [],
                "context_used": False,
                "error": f"Knowledge base error: {str(e)}",
                "fallback_used": True
            }
        except Exception as fallback_error:
            logger.error("Fallback response failed: %s", str(fallback_error), exc_info=True)
            return {"error": f"Complete system error: {str(e)}"}


@app.get("/collection/info")
async def get_collection_info(pipeline: EmbeddingPipeline = Depends(get_embedding_pipeline)):
    """Get information about the document collection."""
    try:
        info = pipeline.vector_store.get_collection_info()
        return JSONResponse(content=info)
    except Exception as e:
        # Avoid passing unsupported keyword arguments to the stdlib logger
        logger.error(
            "Failed to get collection info: %s",
            str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"Failed to get collection info: {str(e)}")


@app.get("/search")
async def search_documents(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(10, ge=1, le=50, description="Number of results to return"),
    similarity_threshold: float = Query(SIMILARITY_DEFAULT, ge=0.0, le=1.0, description="Minimum similarity threshold"),
    include_scores: bool = Query(True, description="Include similarity scores in response"),
    pipeline: EmbeddingPipeline = Depends(get_embedding_pipeline)
):
    """Enhanced search for similar documents with similarity filtering."""
    try:
        # Ensure similarity_threshold is properly converted to float
        threshold_val = float(similarity_threshold)
        
        results = pipeline.query_embeddings(query, n_results=top_k)
        
        if not results.get('documents') or not results['documents'][0]:
            return JSONResponse(content={
                "query": query,
                "results": [],
                "count": 0,
                "similarity_threshold": threshold_val,
                "total_documents_searched": 0
            })
        
        # Format and filter results
        documents = results['documents'][0]
        metadatas = results['metadatas'][0] if results.get('metadatas') else [{}] * len(documents)
        distances = results['distances'][0] if results.get('distances') else [1.0] * len(documents)
        
        formatted_results = []
        for doc, metadata, distance in zip(documents, metadatas, distances):
            # Chroma may return distances as Decimal or str; coerce to float
            try:
                distance_val = float(distance)
            except (TypeError, ValueError):
                logger.warning("Invalid distance value from Chroma - value: %s, type: %s", str(distance), type(distance).__name__)
                continue
            
            # Convert distance → similarity
            similarity_score = distance_to_similarity(distance_val)
            
            # Apply similarity threshold filter
            if similarity_score >= threshold_val:
                result = {
                    'content': doc,
                    'metadata': metadata,
                    'content_preview': doc[:200] + "..." if len(doc) > 200 else doc
                }
                
                if include_scores:
                    result['similarity_score'] = similarity_score
                    result['distance'] = float(distance_val)
                
                formatted_results.append(result)
        
        return JSONResponse(content={
            "query": query,
            "results": formatted_results,
            "count": len(formatted_results),
            "similarity_threshold": threshold_val,
            "total_documents_searched": len(documents),
            "filtered_by_threshold": len(documents) - len(formatted_results)
        })
        
    except Exception as e:
        # Avoid passing unsupported keyword arguments to the stdlib logger
        logger.error(
            "Search failed for query '%s': %s",
            query[:50],
            str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/similarity/validate")
async def validate_similarity_search(
    query: str = Query(..., description="Test query for similarity validation"),
    test_thresholds: List[float] = Query([0.1, 0.3, 0.5, 0.7, 0.9], description="Similarity thresholds to test"),
    top_k: int = Query(10, ge=1, le=50, description="Number of results to retrieve"),
    pipeline: EmbeddingPipeline = Depends(get_embedding_pipeline)
):
    """Validate similarity search functionality across different thresholds."""
    try:
        # Get raw results
        results = pipeline.query_embeddings(query, n_results=top_k)
        
        if not results.get('documents') or not results['documents'][0]:
            return JSONResponse(content={
                "query": query,
                "validation_results": [],
                "total_documents": 0,
                "error": "No documents found in knowledge base"
            })
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0] if results.get('metadatas') else [{}] * len(documents)
        distances = results['distances'][0] if results.get('distances') else [1.0] * len(documents)
        
        # Calculate similarity scores with proper type conversion
        similarity_scores = []
        for distance in distances:
            try:
                distance_val = float(distance)
                similarity_scores.append(distance_to_similarity(distance_val))
            except (TypeError, ValueError):
                logger.warning("Invalid distance value from Chroma - value: %s, type: %s", str(distance), type(distance).__name__)
                continue
        
        # Test each threshold
        validation_results = []
        for threshold in test_thresholds:
            threshold_val = float(threshold)
            relevant_count = sum(1 for score in similarity_scores if score >= threshold_val)
            avg_similarity = sum(score for score in similarity_scores if score >= threshold_val) / max(relevant_count, 1)
            
            validation_results.append({
                "threshold": threshold_val,
                "relevant_documents": relevant_count,
                "filtered_documents": len(documents) - relevant_count,
                "average_similarity": float(avg_similarity) if relevant_count > 0 else 0.0,
                "percentage_relevant": float(relevant_count / len(documents) * 100) if documents else 0.0
            })
        
        # Calculate overall statistics
        min_similarity = min(similarity_scores) if similarity_scores else 0.0
        max_similarity = max(similarity_scores) if similarity_scores else 0.0
        avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
        
        return JSONResponse(content={
            "query": query,
            "validation_results": validation_results,
            "total_documents": len(documents),
            "similarity_statistics": {
                "min_similarity": float(min_similarity),
                "max_similarity": float(max_similarity),
                "average_similarity": float(avg_similarity),
                "similarity_scores": similarity_scores
            },
            "recommended_threshold": float(max(0.3, avg_similarity - 0.2)),  # Conservative recommendation
            "quality_assessment": {
                "excellent": sum(1 for score in similarity_scores if score >= 0.8),
                "good": sum(1 for score in similarity_scores if 0.6 <= score < 0.8),
                "fair": sum(1 for score in similarity_scores if 0.4 <= score < 0.6),
                "poor": sum(1 for score in similarity_scores if score < 0.4)
            }
        })
        
    except Exception as e:
        # Avoid passing unsupported keyword arguments to the stdlib logger
        logger.error(
            "Similarity validation failed for query '%s': %s",
            query[:50],
            str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"Similarity validation failed: {str(e)}")


@app.get("/metrics")
async def get_metrics():
    """Get basic metrics for monitoring."""
    try:
        pipeline = get_embedding_pipeline()
        collection_info = pipeline.vector_store.get_collection_info()
        
        # Check service health
        ollama_healthy = await ollama_client.health_check()
        
        metrics_text = f"""# HELP collection_document_count Number of documents in collection
# TYPE collection_document_count gauge
collection_document_count {collection_info.get("count", 0)}

# HELP ollama_status Ollama service status (1=healthy, 0=unhealthy)
# TYPE ollama_status gauge
ollama_status {1 if ollama_healthy else 0}

# HELP api_timestamp Current timestamp
# TYPE api_timestamp gauge
api_timestamp {int(datetime.utcnow().timestamp())}
"""
        
        return Response(content=metrics_text, media_type="text/plain")
        
    except Exception as e:
        # Avoid passing unsupported keyword arguments to the stdlib logger
        logger.error(
            "Failed to get metrics: %s",
            str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@app.get("/prometheus")
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    registry = get_metrics_registry()
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


@app.middleware("http")
async def metrics_middleware(request, call_next):
    """Middleware to track request metrics."""
    start_time = time.time()
    
    # Track active connections
    get_active_connections().inc()
    
    try:
        response = await call_next(request)
        
        # Track request metrics
        duration = time.time() - start_time
        track_request_metrics(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code,
            duration=duration
        )
        
        return response
    finally:
        get_active_connections().dec()


@app.get("/ask/stream")
async def ask_question_stream(
    question: str = Query(..., description="Question to ask the RAG system"),
    top_k: int = Query(5, ge=1, le=20, description="Number of similar documents to retrieve"),
    similarity_threshold: float = Query(SIMILARITY_DEFAULT, ge=0.0, le=1.0, description="Minimum similarity threshold for relevant documents"),
    pipeline: EmbeddingPipeline = Depends(get_embedding_pipeline)
):
    """Enhanced streaming RAG endpoint with fallback logic and similarity thresholds."""
    if not config.api.streaming_enabled:
        raise HTTPException(
            status_code=400,
            detail="Streaming is not enabled. Set API_STREAMING_ENABLED=true to enable."
        )
    
    async def generate_stream():
        """Generate the enhanced streaming response with fallback logic."""
        start_time = datetime.utcnow()
        
        try:
            # Ensure similarity_threshold is properly converted to float
            logger.info("Starting streaming RAG query",
                       question=question[:50],
                       similarity_threshold=similarity_threshold,
                       similarity_threshold_type=type(similarity_threshold).__name__,
                       top_k=top_k)
            
            threshold_val = float(similarity_threshold)
            logger.debug("Converted threshold to float",
                        threshold_val=threshold_val,
                        threshold_val_type=type(threshold_val).__name__)
            
            # Send initial metadata
            yield "data: " + json.dumps({
                "type": "metadata",
                "timestamp": datetime.utcnow().isoformat(),
                "question": question,
                "similarity_threshold": threshold_val,
                "top_k": top_k
            }) + "\n\n"
            
            # Step 1: Retrieve similar documents
            try:
                logger.info("About to call pipeline.query_embeddings")
                query_results = pipeline.query_embeddings(question, n_results=top_k)
                logger.info("Successfully called pipeline.query_embeddings")
                
                # Debug log the query results structure
                logger.info("Query results received",
                           has_documents=bool(query_results.get('documents')),
                           has_distances=bool(query_results.get('distances')),
                           has_metadatas=bool(query_results.get('metadatas')))
                
                if query_results.get('distances'):
                    distances_sample = query_results['distances'][0][:3] if query_results['distances'][0] else []
                    logger.info("Distance sample from ChromaDB",
                               sample_distances=distances_sample,
                               sample_types=[type(d).__name__ for d in distances_sample])
            except Exception as query_error:
                logger.error(
                    "Error in pipeline.query_embeddings: %s (%s)",
                    str(query_error),
                    type(query_error).__name__,
                    exc_info=True,
                )
                raise
            
            # Step 2: Filter results by similarity threshold
            relevant_docs = []
            sources = []
            use_fallback = False
            
            if query_results.get('documents') and query_results['documents'][0]:
                documents = query_results['documents'][0]
                metadatas = query_results['metadatas'][0] if query_results.get('metadatas') else [{}] * len(documents)
                distances = query_results['distances'][0] if query_results.get('distances') else [1.0] * len(documents)
                
                # Debug logging for distance types
                logger.debug("Processing distances from ChromaDB",
                           distance_count=len(distances),
                           distance_types=[type(d).__name__ for d in distances[:3]],
                           distance_values=[str(d) for d in distances[:3]])
                
                for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
                    # Chroma may return distances as Decimal or str; coerce to float
                    try:
                        # More robust distance conversion
                        if distance is None:
                            logger.warning("Null distance value from Chroma - index: %d", i)
                            continue
                        
                        # Handle different types that ChromaDB might return
                        if isinstance(distance, str):
                            distance_val = float(distance)
                        elif hasattr(distance, '__float__'):
                            distance_val = float(distance)
                        else:
                            distance_val = float(str(distance))
                            
                    except (TypeError, ValueError, AttributeError) as e:
                        logger.error(
                            "Failed to convert distance value at index %s (%s): %s",
                            i,
                            type(distance).__name__,
                            str(e),
                            exc_info=True,
                        )
                        continue
                    
                    # Convert distance → similarity
                    try:
                        similarity_score = distance_to_similarity(distance_val)
                    except (TypeError, ValueError) as e:
                        logger.error(
                            "Failed to calculate similarity score for distance %s: %s",
                            distance_val,
                            str(e),
                            exc_info=True,
                        )
                        continue
                    
                    # Debug the comparison types
                    logger.debug("Comparing similarity score",
                               similarity_score=similarity_score,
                               similarity_score_type=type(similarity_score).__name__,
                               threshold_val=threshold_val,
                               threshold_val_type=type(threshold_val).__name__)
                    
                    try:
                        if similarity_score >= threshold_val:
                            relevant_docs.append({
                                'document': doc,
                                'metadata': metadata
                            })
                            
                            sources.append({
                                'chunk_id': metadata.get('source_id', f'chunk_{i}'),
                                'title': metadata.get('title', 'Unknown'),
                                'url': metadata.get('url', 'Unknown'),
                                'content_type': metadata.get('content_type', 'documentation'),
                                'similarity_score': float(similarity_score),
                                'chunk_text': doc[:200] + "..." if len(doc) > 200 else doc
                            })
                    except (TypeError, ValueError) as e:
                        logger.error(
                            "Failed to process similarity comparison (score=%s, threshold=%s): %s",
                            similarity_score,
                            threshold_val,
                            str(e),
                            exc_info=True,
                        )
                        continue
                
                # Send search results info
                yield "data: " + json.dumps({
                    "type": "search_results",
                    "total_documents_found": len(documents),
                    "relevant_documents_count": len(relevant_docs),
                    "similarity_threshold": threshold_val
                }) + "\n\n"
                
                if not relevant_docs:
                    use_fallback = True
                    yield "data: " + json.dumps({
                        "type": "info",
                        "message": f"No documents found above similarity threshold {threshold_val}. Using general knowledge."
                    }) + "\n\n"
            else:
                use_fallback = True
                yield "data: " + json.dumps({
                    "type": "info",
                    "message": "No documents found in knowledge base. Using general knowledge."
                }) + "\n\n"
            
            # Send sources (if any)
            yield "data: " + json.dumps({
                "type": "sources",
                "sources": sources,
                "context_used": not use_fallback
            }) + "\n\n"
            
            # Step 3: Generate appropriate prompt
            if use_fallback:
                rag_prompt = f"""I don't have specific documents in my knowledge base that are relevant to your question (similarity threshold: {threshold_val}). However, I can provide a general answer based on my training:

Question: {question}

Please provide a helpful answer based on general knowledge, and mention that this response is not based on specific documents in the knowledge base."""
            else:
                context = "\n\n".join([doc['document'] for doc in relevant_docs])
                rag_prompt = f"""Based on the following context, please answer the question. If the context doesn't contain enough information to fully answer the question, please say so and provide what information you can.

Context: {context}

Question: {question}

Answer:"""
            
            # Step 4: Stream LLM response
            yield "data: " + json.dumps({
                "type": "answer_start",
                "using_fallback": use_fallback
            }) + "\n\n"
            
            full_answer = ""
            
            # Use the enhanced LLM client for streaming
            try:
                messages = [{"role": "user", "content": rag_prompt}]
                stream_response = get_completion(messages, stream=True)
                
                if hasattr(stream_response, '__iter__'):
                    for chunk in stream_response:
                        if chunk:
                            full_answer += chunk
                            yield "data: " + json.dumps({
                                "type": "answer_chunk",
                                "chunk": chunk
                            }) + "\n\n"
                else:
                    # Fallback for non-streaming response
                    full_answer = str(stream_response)
                    yield "data: " + json.dumps({
                        "type": "answer_chunk",
                        "chunk": full_answer
                    }) + "\n\n"
                    
            except Exception as stream_error:
                logger.warning("Streaming failed, using non-streaming response - error: %s", str(stream_error))
                full_answer = get_completion(rag_prompt, stream=False)
                yield "data: " + json.dumps({
                    "type": "answer_chunk",
                    "chunk": full_answer
                }) + "\n\n"
            
            # Send completion
            response_time = (datetime.utcnow() - start_time).total_seconds()
            yield "data: " + json.dumps({
                "type": "answer_complete",
                "response_time": response_time,
                "full_answer": full_answer,
                "context_used": not use_fallback,
                "sources_count": len(sources)
            }) + "\n\n"
            
            logger.info("Enhanced streaming RAG query completed",
                       question=question[:50],
                       sources_count=len(sources),
                       context_used=not use_fallback,
                       response_time=response_time)
            
        except Exception as e:
            logger.error(
                "Streaming RAG query failed for question '%s': %s",
                question[:50],
                str(e),
                exc_info=True,
            )
            
            # Provide fallback response even on error
            try:
                yield "data: " + json.dumps({
                    "type": "error_fallback",
                    "message": "Error accessing knowledge base, providing general response"
                }) + "\n\n"
                
                fallback_prompt = f"""There was an error accessing the knowledge base. Please provide a helpful general answer to this question:

Question: {question}

Note: This response is based on general knowledge only, not specific documents."""
                
                fallback_answer = get_completion(fallback_prompt, stream=False)
                yield "data: " + json.dumps({
                    "type": "answer_chunk",
                    "chunk": fallback_answer
                }) + "\n\n"
                
                yield "data: " + json.dumps({
                    "type": "answer_complete",
                    "full_answer": fallback_answer,
                    "context_used": False,
                    "error": str(e),
                    "fallback_used": True
                }) + "\n\n"
                
            except Exception as fallback_error:
                logger.error("Fallback response failed: %s", str(fallback_error), exc_info=True)
                yield "data: " + json.dumps({
                    "type": "error",
                    "message": f"Complete system error: {str(e)}"
                }) + "\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "X-Accel-Buffering": "no"  # Disable Nginx buffering
        }
    )


@app.get("/ask_stream")
def ask_stream(question: str, top_k: int = 3):
    """
    RAG streaming endpoint: embed question, retrieve sources from ChromaDB, stream answer using llm_client.
    """
    import logging
    logger = logging.getLogger(__name__)
    try:
        # 1. Embed the question using llm_client
        question_embedding = get_embedding(question)
        # 2. Retrieve top_k sources from ChromaDB
        pipeline = EmbeddingPipeline()
        results = pipeline.vector_store.query_similar(
            query_embedding=question_embedding,
            n_results=top_k
        )
        context_docs = results['documents'][0] if results['documents'] else []
        context = "\n\n".join(context_docs)
        # 3. Stream answer using llm_client
        prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
        return StreamingResponse(get_completion(prompt, stream=True), media_type="text/plain")
    except Exception as e:
        logger.error(f"RAG streaming API error: {e}")
        return {"error": str(e)}


@app.get("/")
async def serve_chat_interface():
    """Serve the chat interface HTML page."""
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    index_path = os.path.join(static_dir, "index.html")
    
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        return JSONResponse(
            content={"message": "Chat interface not available. Static files not found."},
            status_code=404
        )


@app.get("/chat")
async def serve_chat_interface_alt():
    """Alternative route for the chat interface."""
    return await serve_chat_interface()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=config.api.host if hasattr(config, 'api') else "0.0.0.0",
        port=config.api.port if hasattr(config, 'api') else 8001,
        reload=True
    )