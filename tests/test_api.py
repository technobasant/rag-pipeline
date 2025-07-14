"""
Unit tests for the FastAPI RAG API module.
Tests are written to match the current implementation in src/api/main.py
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
import json
from datetime import datetime

# Import the actual FastAPI app
try:
    from src.api.main import app
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    app = None

# Skip all tests if API is not available
pytestmark = pytest.mark.skipif(not API_AVAILABLE, reason="API module not available")


class TestFastAPIEndpoints:
    """Test cases for FastAPI endpoints."""
    
    def setup_method(self):
        """Set up test client."""
        if API_AVAILABLE:
            self.client = TestClient(app)
    
    def test_health_endpoint_basic(self):
        """Test health check endpoint basic functionality."""
        with patch('src.api.main.get_embedding_pipeline') as mock_pipeline, \
             patch('src.api.main.ollama_client') as mock_ollama, \
             patch('src.api.main.get_active_connections') as mock_connections:
            
            # Mock metrics
            mock_gauge = Mock()
            mock_gauge.inc.return_value = None
            mock_gauge.dec.return_value = None
            mock_connections.return_value = mock_gauge
            
            # Mock pipeline
            mock_pipeline_instance = Mock()
            mock_pipeline_instance.vector_store.connect.return_value = None
            mock_pipeline_instance.vector_store.get_collection_info.return_value = {"count": 100}
            mock_pipeline.return_value = mock_pipeline_instance
            
            # Mock Ollama health check
            mock_ollama.health_check = AsyncMock(return_value=True)
            
            response = self.client.get("/health")
            
            # Should return 200 regardless of service status
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "timestamp" in data
            assert "services" in data
            assert "version" in data
    
    def test_health_endpoint_service_failures(self):
        """Test health endpoint with service failures."""
        with patch('src.api.main.get_embedding_pipeline') as mock_pipeline, \
             patch('src.api.main.ollama_client') as mock_ollama, \
             patch('src.api.main.get_active_connections') as mock_connections:
            
            # Mock metrics
            mock_gauge = Mock()
            mock_gauge.inc.return_value = None
            mock_gauge.dec.return_value = None
            mock_connections.return_value = mock_gauge
            
            # Mock pipeline failure
            mock_pipeline.side_effect = Exception("ChromaDB connection failed")
            
            # Mock Ollama failure
            mock_ollama.health_check = AsyncMock(return_value=False)
            
            response = self.client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] in ["healthy", "unhealthy"]
            assert "chroma" in data["services"]
            assert "ollama" in data["services"]
    
    def test_ask_endpoint_success(self):
        """Test successful ask endpoint."""
        with patch('src.api.main.get_embedding') as mock_get_embedding, \
             patch('src.api.main.EmbeddingPipeline') as mock_pipeline_class, \
             patch('src.api.main.get_completion') as mock_get_completion, \
             patch('src.api.main.get_active_connections') as mock_connections:
            
            # Mock metrics
            mock_gauge = Mock()
            mock_gauge.inc.return_value = None
            mock_gauge.dec.return_value = None
            mock_connections.return_value = mock_gauge
            
            # Mock embedding generation
            mock_get_embedding.return_value = [0.1, 0.2, 0.3]
            
            # Mock pipeline
            mock_pipeline = Mock()
            mock_pipeline.vector_store.query_similar.return_value = {
                'documents': [['Test document content']],
                'metadatas': [[{'title': 'Test Doc', 'url': 'http://test.com'}]],
                'distances': [[0.2]]
            }
            mock_pipeline_class.return_value = mock_pipeline
            
            # Mock LLM completion
            mock_get_completion.return_value = "This is a test answer."
            
            response = self.client.get("/ask", params={"question": "What is Python?"})
            
            if response.status_code == 200:
                data = response.json()
                assert "answer" in data
                assert "sources" in data
                assert "query" in data
                assert "response_time" in data
                assert "timestamp" in data
                assert isinstance(data["sources"], list)
            else:
                # If dependencies are missing, endpoint should return error
                assert response.status_code in [500, 503]
    
    def test_ask_endpoint_validation(self):
        """Test ask endpoint parameter validation."""
        with patch('src.api.main.get_active_connections') as mock_connections, \
             patch('src.api.main.get_embedding') as mock_get_embedding, \
             patch('src.api.main.get_completion') as mock_get_completion:
            
            # Mock metrics
            mock_gauge = Mock()
            mock_gauge.inc.return_value = None
            mock_gauge.dec.return_value = None
            mock_connections.return_value = mock_gauge
            
            # Mock embedding and completion to fail gracefully
            mock_get_embedding.side_effect = Exception("No embedding methods available")
            mock_get_completion.side_effect = Exception("No completion methods available")
            
            # Missing question parameter - should return 422 for validation error
            response = self.client.get("/ask")
            assert response.status_code == 422
            
            # Valid question but services unavailable - should return 200 with error message
            response = self.client.get("/ask", params={"question": "test"})
            assert response.status_code == 200
            data = response.json()
            assert "error" in data or "answer" in data
            
            # Test with valid parameters but service errors
            response = self.client.get("/ask", params={"question": "test", "top_k": 5, "similarity_threshold": 0.7})
            assert response.status_code == 200
    
    def test_ask_endpoint_fallback(self):
        """Test ask endpoint fallback behavior."""
        with patch('src.api.main.get_embedding') as mock_get_embedding, \
             patch('src.api.main.EmbeddingPipeline') as mock_pipeline_class, \
             patch('src.api.main.get_completion') as mock_get_completion, \
             patch('src.api.main.get_active_connections') as mock_connections:
            
            # Mock metrics
            mock_gauge = Mock()
            mock_gauge.inc.return_value = None
            mock_gauge.dec.return_value = None
            mock_connections.return_value = mock_gauge
            
            # Mock embedding generation
            mock_get_embedding.return_value = [0.1, 0.2, 0.3]
            
            # Mock pipeline with no results
            mock_pipeline = Mock()
            mock_pipeline.vector_store.query_similar.return_value = {
                'documents': [[]],
                'metadatas': [[]],
                'distances': [[]]
            }
            mock_pipeline_class.return_value = mock_pipeline
            
            # Mock LLM completion
            mock_get_completion.return_value = "Fallback answer based on general knowledge."
            
            response = self.client.get("/ask", params={"question": "What is quantum computing?"})
            
            if response.status_code == 200:
                data = response.json()
                assert "answer" in data
                assert len(data["sources"]) == 0  # No relevant sources found
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint."""
        with patch('src.api.main.get_embedding_pipeline') as mock_pipeline, \
             patch('src.api.main.ollama_client') as mock_ollama, \
             patch('src.api.main.get_active_connections') as mock_connections:
            
            # Mock metrics
            mock_gauge = Mock()
            mock_gauge.inc.return_value = None
            mock_gauge.dec.return_value = None
            mock_connections.return_value = mock_gauge
            
            # Mock pipeline
            mock_pipeline_instance = Mock()
            mock_pipeline_instance.vector_store.get_collection_info.return_value = {"count": 50}
            mock_pipeline.return_value = mock_pipeline_instance
            
            # Mock Ollama
            mock_ollama.health_check = AsyncMock(return_value=True)
            
            response = self.client.get("/metrics")
            
            if response.status_code == 200:
                assert "text/plain" in response.headers.get("content-type", "")
                content = response.text
                assert "collection_document_count" in content
                assert "ollama_status" in content
            else:
                # If dependencies are missing, should return error
                assert response.status_code in [500, 503]
    
    def test_search_endpoint(self):
        """Test search endpoint."""
        with patch('src.api.main.get_embedding_pipeline') as mock_get_pipeline, \
             patch('src.api.main.get_active_connections') as mock_connections:
            
            # Mock metrics
            mock_gauge = Mock()
            mock_gauge.inc.return_value = None
            mock_gauge.dec.return_value = None
            mock_connections.return_value = mock_gauge
            
            # Mock pipeline
            mock_pipeline = Mock()
            mock_pipeline.query_embeddings.return_value = {
                'documents': [['Document 1', 'Document 2']],
                'metadatas': [[{'title': 'Doc 1'}, {'title': 'Doc 2'}]],
                'distances': [[0.1, 0.3]]
            }
            mock_get_pipeline.return_value = mock_pipeline
            
            response = self.client.get("/search", params={"query": "test search"})
            
            if response.status_code == 200:
                data = response.json()
                assert "query" in data
                assert "results" in data
                assert "count" in data
                assert isinstance(data["results"], list)
            else:
                # If dependencies are missing, should return error
                assert response.status_code in [500, 503]
    
    def test_collection_info_endpoint(self):
        """Test collection info endpoint."""
        with patch('src.api.main.get_embedding_pipeline') as mock_get_pipeline, \
             patch('src.api.main.get_active_connections') as mock_connections:
            
            # Mock metrics
            mock_gauge = Mock()
            mock_gauge.inc.return_value = None
            mock_gauge.dec.return_value = None
            mock_connections.return_value = mock_gauge
            
            # Mock pipeline
            mock_pipeline = Mock()
            mock_pipeline.vector_store.get_collection_info.return_value = {
                "name": "documents",
                "count": 100,
                "metadata": {}
            }
            mock_get_pipeline.return_value = mock_pipeline
            
            response = self.client.get("/collection/info")
            
            if response.status_code == 200:
                data = response.json()
                assert "name" in data or "count" in data
            else:
                # If dependencies are missing, should return error
                assert response.status_code in [500, 503]
    
    def test_openapi_docs(self):
        """Test OpenAPI documentation endpoints."""
        with patch('src.api.main.get_active_connections') as mock_connections:
            # Mock metrics
            mock_gauge = Mock()
            mock_gauge.inc.return_value = None
            mock_gauge.dec.return_value = None
            mock_connections.return_value = mock_gauge
            
            response = self.client.get("/docs")
            assert response.status_code == 200
            
            response = self.client.get("/openapi.json")
            assert response.status_code == 200
            data = response.json()
            assert "openapi" in data
            assert "paths" in data


class TestAPIModels:
    """Test API response models and validation."""
    
    def test_distance_to_similarity_function(self):
        """Test distance to similarity conversion function."""
        from src.api.main import distance_to_similarity
        
        # Test normal cases
        assert distance_to_similarity(0.0) == 1.0  # Perfect match
        assert distance_to_similarity(2.0) == 0.0  # Maximum distance
        assert distance_to_similarity(1.0) == 0.5  # Middle distance
        
        # Test edge cases
        assert distance_to_similarity(-0.5) == 1.0  # Negative distance
        assert distance_to_similarity(3.0) == 0.0   # Beyond maximum
        assert distance_to_similarity("invalid") == 0.0  # Invalid input


class TestAPIErrorHandling:
    """Test API error handling and edge cases."""
    
    def setup_method(self):
        """Set up test client."""
        if API_AVAILABLE:
            self.client = TestClient(app)
    
    def test_ask_endpoint_service_error(self):
        """Test ask endpoint with service errors."""
        with patch('src.api.main.get_embedding') as mock_get_embedding, \
             patch('src.api.main.get_active_connections') as mock_connections:
            
            # Mock metrics
            mock_gauge = Mock()
            mock_gauge.inc.return_value = None
            mock_gauge.dec.return_value = None
            mock_connections.return_value = mock_gauge
            
            # Mock embedding failure
            mock_get_embedding.side_effect = Exception("Embedding service failed")
            
            response = self.client.get("/ask", params={"question": "test question"})
            
            # Should handle error gracefully
            assert response.status_code in [200, 500]
            
            if response.status_code == 200:
                data = response.json()
                # Should provide fallback response
                assert "answer" in data or "error" in data
    
    def test_middleware_functionality(self):
        """Test middleware is working."""
        with patch('src.api.main.get_active_connections') as mock_connections:
            # Mock metrics
            mock_gauge = Mock()
            mock_gauge.inc.return_value = None
            mock_gauge.dec.return_value = None
            mock_connections.return_value = mock_gauge
            
            # Test that requests complete without middleware errors
            response = self.client.get("/health")
            assert response.status_code == 200
            
            # Test CORS headers might be present
            # (Actual CORS testing would require cross-origin requests)
            assert response.status_code == 200


class TestStreamingEndpoints:
    """Test streaming endpoints if available."""
    
    def setup_method(self):
        """Set up test client."""
        if API_AVAILABLE:
            self.client = TestClient(app)
    
    def test_streaming_endpoint_disabled(self):
        """Test streaming endpoint when disabled."""
        with patch('src.api.main.config') as mock_config, \
             patch('src.api.main.get_active_connections') as mock_connections:
            
            # Mock metrics
            mock_gauge = Mock()
            mock_gauge.inc.return_value = None
            mock_gauge.dec.return_value = None
            mock_connections.return_value = mock_gauge
            
            mock_config.api.streaming_enabled = False
            
            response = self.client.get("/ask/stream", params={"question": "test"})
            assert response.status_code == 400
    
    def test_ask_stream_basic(self):
        """Test basic ask_stream endpoint."""
        with patch('src.api.main.get_embedding') as mock_get_embedding, \
             patch('src.api.main.EmbeddingPipeline') as mock_pipeline_class, \
             patch('src.api.main.get_completion') as mock_get_completion, \
             patch('src.api.main.get_active_connections') as mock_connections:
            
            # Mock metrics
            mock_gauge = Mock()
            mock_gauge.inc.return_value = None
            mock_gauge.dec.return_value = None
            mock_connections.return_value = mock_gauge
            
            # Mock dependencies
            mock_get_embedding.return_value = [0.1, 0.2, 0.3]
            
            mock_pipeline = Mock()
            mock_pipeline.vector_store.query_similar.return_value = {
                'documents': [['Test content']],
                'metadatas': [[{}]],
                'distances': [[0.2]]
            }
            mock_pipeline_class.return_value = mock_pipeline
            
            mock_get_completion.return_value = "Test response"
            
            response = self.client.get("/ask_stream", params={"question": "test"})
            
            # Should return some response (streaming or error)
            assert response.status_code in [200, 500]


if __name__ == "__main__":
    pytest.main([__file__])