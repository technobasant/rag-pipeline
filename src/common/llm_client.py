"""
LLM client module for unified LLM operations using LiteLLM.

This module provides a unified interface for both embeddings and text generation
using LiteLLM, supporting Ollama, OpenAI, and other providers.
"""

import os
import logging
from typing import List, Optional, Dict, Any, Union
import structlog

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    logging.warning("LiteLLM not available, falling back to basic implementations")

try:
    import ollama
    from ollama import AsyncClient
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    AsyncClient = None

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = structlog.get_logger(__name__)

# Configuration from environment
LITELLM_API_BASE = os.getenv("LITELLM_API_BASE", "http://localhost:11434")
LITELLM_MODEL = os.getenv("LITELLM_MODEL", "ollama/qwen3:0.6b")
# Default embedding model
# Note: do NOT include the "ollama/" prefix – the Ollama CLI/API expects the raw
# model name (e.g. "mahonzhan/all-MiniLM-L6-v2").  The user can still override
# via environment variable if they have a different local model.
LITELLM_EMBED_MODEL = os.getenv("LITELLM_EMBED_MODEL", "mahonzhan/all-MiniLM-L6-v2")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Global model cache
_embedding_model_cache = {}


def get_embedding(
    text: str, 
    model: Optional[str] = None,
    **kwargs
) -> List[float]:
    """
    Generate embeddings for text using LiteLLM or fallback methods.
    
    Args:
        text: Input text to embed
        model: Model name (optional, uses default if not provided)
        **kwargs: Additional arguments for the embedding model
        
    Returns:
        List of embedding values as floats
        
    Raises:
        Exception: If embedding generation fails
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    
    model = model or LITELLM_EMBED_MODEL
    
    # Try LiteLLM first if available
    if LITELLM_AVAILABLE:
        try:
            logger.debug("Using LiteLLM for embeddings", model=model, text_length=len(text))
            
            # Configure LiteLLM
            if LITELLM_API_KEY:
                litellm.api_key = LITELLM_API_KEY
            if LITELLM_API_BASE:
                litellm.api_base = LITELLM_API_BASE
            
            response = litellm.embedding(
                model=model,
                input=[text],
                **kwargs
            )
            
            if response and response.data and len(response.data) > 0:
                embedding = response.data[0].embedding
                logger.debug("LiteLLM embedding successful", 
                           model=model, 
                           embedding_dim=len(embedding))
                return embedding
            else:
                raise Exception("Empty response from LiteLLM")
                
        except Exception as e:
            # If LiteLLM was called without a configured provider it raises a
            # BadRequestError with the message "LLM Provider NOT provided".
            # That scenario is expected in setups that rely exclusively on
            # Ollama or local back-ends, so we treat it as an INFO/DEBUG path
            # instead of polluting logs with warnings.
            if "LLM Provider NOT provided" in str(e):
                logger.debug("LiteLLM provider not configured – falling back", 
                             error=str(e), model=model)
            else:
                logger.warning("LiteLLM embedding failed, trying fallback", 
                               error=str(e), model=model)
    
    # Fallback to Ollama if available
    if OLLAMA_AVAILABLE:
        try:
            # Ollama models are referenced without any registry prefix.
            # Strip "ollama/" if the user accidentally included it.
            ollama_model = model.replace("ollama/", "") if model else model

            logger.debug("Using Ollama for embeddings", model=ollama_model)
            
            response = ollama.embed(
                model=ollama_model,
                input=text
            )
            
            if response and 'embeddings' in response and len(response['embeddings']) > 0:
                embedding = response['embeddings'][0]
                logger.debug("Ollama embedding successful",
                           model=ollama_model,
                           embedding_dim=len(embedding))
                return embedding
            else:
                raise Exception("Invalid response from Ollama")
                
        except Exception as e:
            logger.warning("Ollama embedding failed, trying SentenceTransformers",
                         error=str(e), model=ollama_model)
    
    # # Final fallback to SentenceTransformers
    # if SENTENCE_TRANSFORMERS_AVAILABLE:
    #     try:
    #         # Use sentence-transformers model name if available
    #         st_model = EMBEDDING_MODEL if "sentence-transformers" in EMBEDDING_MODEL else "sentence-transformers/all-MiniLM-L6-v2"
            
    #         logger.debug("Using SentenceTransformers for embeddings", model=st_model)
            
    #         # Cache model to avoid reloading
    #         if st_model not in _embedding_model_cache:
    #             _embedding_model_cache[st_model] = SentenceTransformer(st_model)
            
    #         model_instance = _embedding_model_cache[st_model]
    #         embedding = model_instance.encode(text, convert_to_tensor=False).tolist()
            
    #         logger.debug("SentenceTransformers embedding successful", 
    #                    model=st_model, 
    #                    embedding_dim=len(embedding))
    #         return embedding
            
    #     except Exception as e:
    #         logger.error("SentenceTransformers embedding failed", 
    #                    error=str(e), model=st_model)
    #         raise Exception(f"All embedding methods failed. Last error: {str(e)}")
    
    raise Exception("No embedding methods available. Install litellm, ollama, or sentence-transformers")


def get_completion(
    messages: Union[str, List[Dict[str, str]]], 
    model: Optional[str] = None,
    stream: bool = False,
    **kwargs
) -> Union[str, Any]:
    """
    Generate text completion using LiteLLM or fallback methods.
    
    Args:
        messages: Input messages (string or chat format)
        model: Model name (optional, uses default if not provided)
        stream: Whether to stream the response
        **kwargs: Additional arguments for the completion model
        
    Returns:
        Completion text or stream object
        
    Raises:
        Exception: If completion generation fails
    """
    model = model or LITELLM_MODEL
    
    # Convert string to messages format if needed
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    
    # Try LiteLLM first if available
    if LITELLM_AVAILABLE:
        try:
            # Ensure no accidental "ollama/" prefix is sent to the API
            model = model.replace("ollama/", "") if model else model
            logger.debug("Using LiteLLM for completion", model=model, stream=stream)
            
            # Configure LiteLLM
            if LITELLM_API_KEY:
                litellm.api_key = LITELLM_API_KEY
            if LITELLM_API_BASE:
                litellm.api_base = LITELLM_API_BASE
            
            response = litellm.completion(
                model=model,
                messages=messages,
                stream=stream,
                **kwargs
            )
            
            if stream:
                return response
            else:
                content = response.choices[0].message.content
                logger.debug("LiteLLM completion successful", 
                           model=model, 
                           response_length=len(content))
                return content
                
        except Exception as e:
            if "LLM Provider NOT provided" in str(e):
                logger.debug("LiteLLM provider not configured – falling back", 
                             error=str(e), model=model)
            else:
                logger.warning("LiteLLM completion failed, trying fallback", 
                               error=str(e), model=model)
    
    # Fallback to Ollama if available
    if OLLAMA_AVAILABLE:
        try:
            # Ensure no accidental "ollama/" prefix is sent to the API
            model = model.replace("ollama/", "") if model else model
            logger.debug("Using Ollama for completion", model=model, stream=stream)
            
            # Use the chat API for better conversation handling
            response = ollama.chat(
                model=model,
                messages=messages,
                stream=stream
            )
            
            if stream:
                # Return a generator that yields content from streaming response
                def stream_generator():
                    for chunk in response:
                        if 'message' in chunk and 'content' in chunk['message']:
                            yield chunk['message']['content']
                return stream_generator()
            else:
                content = response['message']['content']
                logger.debug("Ollama completion successful",
                           model=model,
                           response_length=len(content))
                return content
                
        except Exception as e:
            logger.error("Ollama completion failed",
                       error=str(e), model=model)
            raise Exception(f"All completion methods failed. Last error: {str(e)}")
    
    raise Exception("No completion methods available. Install litellm or ollama")


def health_check() -> Dict[str, Any]:
    """
    Check health of available LLM services.
    
    Returns:
        Dictionary with health status of each service
    """
    health = {
        "litellm": {"available": LITELLM_AVAILABLE, "status": "unknown"},
        "ollama": {"available": OLLAMA_AVAILABLE, "status": "unknown"},
        "sentence_transformers": {"available": SENTENCE_TRANSFORMERS_AVAILABLE, "status": "unknown"}
    }
    
    # Test LiteLLM
    if LITELLM_AVAILABLE:
        try:
            # Simple test - this might fail but we catch it
            test_response = get_embedding("test", model=LITELLM_EMBED_MODEL)
            health["litellm"]["status"] = "healthy" if test_response else "unhealthy"
        except Exception as e:
            health["litellm"]["status"] = f"unhealthy: {str(e)}"
    
    # Test Ollama
    if OLLAMA_AVAILABLE:
        try:
            ollama.list()  # Simple health check
            health["ollama"]["status"] = "healthy"
        except Exception as e:
            health["ollama"]["status"] = f"unhealthy: {str(e)}"
    
    # Test SentenceTransformers
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            # This is always healthy if imported successfully
            health["sentence_transformers"]["status"] = "healthy"
        except Exception as e:
            health["sentence_transformers"]["status"] = f"unhealthy: {str(e)}"
    
    return health


# Backward compatibility aliases
def embed_texts(texts: List[str], model: Optional[str] = None) -> List[List[float]]:
    """
    Generate embeddings for multiple texts.
    
    Args:
        texts: List of input texts
        model: Model name (optional)
        
    Returns:
        List of embedding vectors
    """
    embeddings = []
    for text in texts:
        try:
            embedding = get_embedding(text, model=model)
            embeddings.append(embedding)
        except Exception as e:
            logger.error("Failed to embed text", text=text[:50], error=str(e))
            embeddings.append([])  # Empty embedding for failed texts
    
    return embeddings


def chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    stream: bool = False,
    **kwargs
) -> Union[str, Any]:
    """
    Chat with the LLM using conversation format.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model: Model name (optional, uses default if not provided)
        stream: Whether to stream the response
        **kwargs: Additional arguments for the chat model
        
    Returns:
        Chat response text or stream object
        
    Raises:
        Exception: If chat generation fails
    """
    return get_completion(messages, model=model, stream=stream, **kwargs)


async def async_chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    stream: bool = False,
    **kwargs
) -> Union[str, Any]:
    """
    Async chat with the LLM using conversation format.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model: Model name (optional, uses default if not provided)
        stream: Whether to stream the response
        **kwargs: Additional arguments for the chat model
        
    Returns:
        Chat response text or async stream object
        
    Raises:
        Exception: If chat generation fails
    """
    model = model or LITELLM_MODEL
    
    # Try Ollama AsyncClient if available
    if OLLAMA_AVAILABLE and AsyncClient:
        try:
            logger.debug("Using Ollama AsyncClient for chat", model=model, stream=stream)
            
            client = AsyncClient()
            response = await client.chat(
                model=model,
                messages=messages,
                stream=stream,
                **kwargs
            )
            
            if stream:
                # Return async generator for streaming
                async def async_stream_generator():
                    async for chunk in response:
                        if 'message' in chunk and 'content' in chunk['message']:
                            yield chunk['message']['content']
                return async_stream_generator()
            else:
                content = response['message']['content']
                logger.debug("Ollama async chat successful",
                           model=model,
                           response_length=len(content))
                return content
                
        except Exception as e:
            logger.error("Ollama async chat failed",
                       error=str(e), model=model)
            raise Exception(f"Async chat failed. Error: {str(e)}")
    
    # Fallback to sync version
    logger.warning("AsyncClient not available, falling back to sync chat")
    return chat(messages, model=model, stream=stream, **kwargs)


def stream_chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Stream chat responses from the LLM.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model: Model name (optional, uses default if not provided)
        **kwargs: Additional arguments for the chat model
        
    Returns:
        Stream generator yielding response chunks
        
    Raises:
        Exception: If streaming fails
    """
    return chat(messages, model=model, stream=True, **kwargs)


async def async_stream_chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Async stream chat responses from the LLM.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model: Model name (optional, uses default if not provided)
        **kwargs: Additional arguments for the chat model
        
    Returns:
        Async stream generator yielding response chunks
        
    Raises:
        Exception: If streaming fails
    """
    return await async_chat(messages, model=model, stream=True, **kwargs)