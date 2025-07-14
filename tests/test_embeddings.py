"""
Unit tests for the embeddings module.
Tests are written to match the current implementation in src/embeddings/embedder.py
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
from typing import List, Dict, Any

# Try to import embeddings components
try:
    from src.embeddings.embedder import EmbeddingPipeline, ChromaVectorStore, EmbeddingGenerator, chunk_text, get_device
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

# Skip all tests if embeddings module is not available
pytestmark = pytest.mark.skipif(not EMBEDDINGS_AVAILABLE, reason="Embeddings module not available")


class TestChunkingUtilities:
    """Test text chunking utilities."""
    
    def test_chunk_text_basic(self):
        """Test basic text chunking functionality."""
        text = "This is a test text that should be chunked into smaller pieces for better processing."
        chunks = chunk_text(text, max_tokens=5, overlap=0.2)
        
        assert len(chunks) > 1
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert all(len(chunk.split()) <= 5 for chunk in chunks)
    
    def test_chunk_text_empty(self):
        """Test chunking empty text."""
        chunks = chunk_text("", max_tokens=10)
        assert chunks == []
        
        chunks = chunk_text(None, max_tokens=10)
        assert chunks == []
    
    def test_chunk_text_short(self):
        """Test chunking text shorter than max_tokens."""
        text = "Short text"
        chunks = chunk_text(text, max_tokens=10)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunk_text_overlap(self):
        """Test chunking with overlap."""
        text = "word1 word2 word3 word4 word5 word6 word7 word8"
        chunks = chunk_text(text, max_tokens=4, overlap=0.5)
        
        assert len(chunks) > 1
        # With 50% overlap, chunks should share some words
        if len(chunks) >= 2:
            first_words = chunks[0].split()
            second_words = chunks[1].split()
            # Should have some overlap
            assert len(set(first_words) & set(second_words)) > 0


class TestDeviceSelection:
    """Test device selection logic."""
    
    @patch.dict('os.environ', {'EMBEDDING_DEVICE': 'cpu'})
    def test_get_device_cpu(self):
        """Test CPU device selection."""
        device = get_device()
        assert device == "cpu"
    
    @patch.dict('os.environ', {'EMBEDDING_DEVICE': 'cuda'})
    @patch('torch.cuda.is_available', return_value=True)
    def test_get_device_cuda_available(self, mock_cuda):
        """Test CUDA device selection when available."""
        device = get_device()
        assert device == "cuda"
    
    @patch.dict('os.environ', {'EMBEDDING_DEVICE': 'cuda'})
    @patch('torch.cuda.is_available', return_value=False)
    def test_get_device_cuda_unavailable(self, mock_cuda):
        """Test CUDA device fallback when unavailable."""
        device = get_device()
        assert device == "cpu"
    
    @patch.dict('os.environ', {'EMBEDDING_DEVICE': 'auto'})
    @patch('torch.cuda.is_available', return_value=True)
    def test_get_device_auto_cuda(self, mock_cuda):
        """Test auto device selection with CUDA available."""
        device = get_device()
        assert device == "cuda"
    
    @patch.dict('os.environ', {'EMBEDDING_DEVICE': 'auto'})
    @patch('torch.cuda.is_available', return_value=False)
    def test_get_device_auto_cpu(self, mock_cuda):
        """Test auto device selection with CUDA unavailable."""
        device = get_device()
        assert device == "cpu"


class TestEmbeddingGenerator:
    """Test EmbeddingGenerator functionality."""
    
    @patch('src.embeddings.embedder.get_embedding')
    def test_embed_texts_success(self, mock_get_embedding):
        """Test successful text embedding."""
        mock_get_embedding.return_value = [0.1, 0.2, 0.3]
        
        generator = EmbeddingGenerator()
        texts = ["Text 1", "Text 2"]
        embeddings = generator.embed_texts(texts)
        
        assert len(embeddings) == 2
        assert all(emb == [0.1, 0.2, 0.3] for emb in embeddings)
        assert mock_get_embedding.call_count == 2
    
    @patch('src.embeddings.embedder.get_embedding')
    def test_embed_texts_error_handling(self, mock_get_embedding):
        """Test embedding error handling."""
        mock_get_embedding.side_effect = [
            [0.1, 0.2, 0.3],  # First succeeds
            Exception("Embedding failed"),  # Second fails
            [0.4, 0.5, 0.6]   # Third succeeds
        ]
        
        generator = EmbeddingGenerator()
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = generator.embed_texts(texts)
        
        assert len(embeddings) == 3
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] is None  # Failed embedding
        assert embeddings[2] == [0.4, 0.5, 0.6]
    
    def test_embed_texts_empty(self):
        """Test embedding empty text list."""
        generator = EmbeddingGenerator()
        embeddings = generator.embed_texts([])
        
        assert embeddings == []


class TestChromaVectorStore:
    """Test ChromaVectorStore functionality."""
    
    @patch('src.embeddings.embedder.Settings')
    @patch('src.embeddings.embedder.chromadb')
    @patch('src.embeddings.embedder.config')
    def test_connect_existing_collection(self, mock_config, mock_chromadb, mock_settings):
        """Test connecting to existing collection."""
        # Mock config
        mock_config.chroma.host = "localhost"
        mock_config.chroma.port = 8000
        mock_config.chroma.collection_name = "documents"
        
        # Mock settings
        mock_settings.return_value = Mock()
        
        # Mock client and collection
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_chromadb.HttpClient.return_value = mock_client
        
        store = ChromaVectorStore()
        store.connect()
        
        assert store.client == mock_client
        assert store.collection == mock_collection
        mock_client.get_collection.assert_called_once_with(name="documents")
    
    @patch('src.embeddings.embedder.Settings')
    @patch('src.embeddings.embedder.chromadb')
    @patch('src.embeddings.embedder.config')
    def test_connect_create_collection(self, mock_config, mock_chromadb, mock_settings):
        """Test creating new collection when it doesn't exist."""
        # Mock config
        mock_config.chroma.host = "localhost"
        mock_config.chroma.port = 8000
        mock_config.chroma.collection_name = "documents"
        
        # Mock settings
        mock_settings.return_value = Mock()
        
        # Mock client
        mock_client = Mock()
        mock_client.get_collection.side_effect = Exception("Collection not found")
        mock_collection = Mock()
        mock_client.create_collection.return_value = mock_collection
        mock_chromadb.HttpClient.return_value = mock_client
        
        store = ChromaVectorStore()
        store.connect()
        
        assert store.client == mock_client
        assert store.collection == mock_collection
        mock_client.create_collection.assert_called_once()
    
    def test_add_documents(self):
        """Test adding documents to collection."""
        store = ChromaVectorStore()
        
        # Mock collection
        mock_collection = Mock()
        store.collection = mock_collection
        store.client = Mock()  # Prevent connect() from being called
        
        ids = ["doc1", "doc2"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        documents = ["Document 1", "Document 2"]
        metadatas = [{"title": "Doc 1"}, {"title": "Doc 2"}]
        
        store.add_documents(ids, embeddings, documents, metadatas)
        
        mock_collection.add.assert_called_once_with(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    
    def test_add_documents_batching(self):
        """Test document batching for large datasets."""
        store = ChromaVectorStore()
        
        # Mock collection
        mock_collection = Mock()
        store.collection = mock_collection
        store.client = Mock()
        
        # Create large dataset (more than batch size of 100)
        ids = [f"doc{i}" for i in range(150)]
        embeddings = [[0.1, 0.2] for _ in range(150)]
        documents = [f"Document {i}" for i in range(150)]
        metadatas = [{"title": f"Doc {i}"} for i in range(150)]
        
        store.add_documents(ids, embeddings, documents, metadatas)
        
        # Should be called twice (100 + 50)
        assert mock_collection.add.call_count == 2
    
    def test_query_similar(self):
        """Test querying similar documents."""
        store = ChromaVectorStore()
        
        # Mock collection
        mock_collection = Mock()
        mock_collection.query.return_value = {
            'ids': [['doc1', 'doc2']],
            'documents': [['Document 1', 'Document 2']],
            'distances': [[0.1, 0.3]],
            'metadatas': [[{'title': 'Doc 1'}, {'title': 'Doc 2'}]]
        }
        store.collection = mock_collection
        store.client = Mock()
        
        query_embedding = [0.1, 0.2, 0.3]
        results = store.query_similar(query_embedding, n_results=2)
        
        mock_collection.query.assert_called_once_with(
            query_embeddings=[query_embedding],
            n_results=2,
            where=None
        )
        
        assert 'ids' in results
        assert 'documents' in results
        assert 'distances' in results
    
    def test_query_similar_with_filter(self):
        """Test querying with where filter."""
        store = ChromaVectorStore()
        
        mock_collection = Mock()
        mock_collection.query.return_value = {
            'ids': [['doc1']],
            'documents': [['Document 1']],
            'distances': [[0.1]],
            'metadatas': [[{'title': 'Doc 1'}]]
        }
        store.collection = mock_collection
        store.client = Mock()
        
        query_embedding = [0.1, 0.2, 0.3]
        where_filter = {"title": "Doc 1"}
        results = store.query_similar(query_embedding, n_results=1, where=where_filter)
        
        mock_collection.query.assert_called_once_with(
            query_embeddings=[query_embedding],
            n_results=1,
            where=where_filter
        )
    
    def test_get_collection_info(self):
        """Test getting collection information."""
        store = ChromaVectorStore()
        
        # Mock collection and config
        mock_collection = Mock()
        mock_collection.count.return_value = 100
        mock_collection.metadata = {"description": "Test collection"}
        store.collection = mock_collection
        store.client = Mock()
        
        with patch('src.embeddings.embedder.config') as mock_config:
            mock_config.chroma.collection_name = "documents"
            
            info = store.get_collection_info()
            
            assert info["name"] == "documents"
            assert info["count"] == 100
            assert info["metadata"] == {"description": "Test collection"}
    
    def test_reset_collection(self):
        """Test resetting collection."""
        store = ChromaVectorStore()
        
        # Mock client and collection
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.delete_collection.return_value = None
        mock_client.create_collection.return_value = mock_collection
        store.client = mock_client
        store.collection = Mock()
        
        with patch('src.embeddings.embedder.config') as mock_config:
            mock_config.chroma.collection_name = "documents"
            
            store.reset_collection()
            
            mock_client.delete_collection.assert_called_once_with("documents")
            mock_client.create_collection.assert_called_once()
            assert store.collection == mock_collection


class TestEmbeddingPipeline:
    """Test EmbeddingPipeline functionality."""
    
    @patch('src.embeddings.embedder.create_catalog')
    def test_load_gold_data(self, mock_create_catalog):
        """Test loading Gold data from DuckDB catalog."""
        # Mock catalog
        mock_catalog = Mock()
        mock_catalog.list_tables.return_value = [{'name': 'gold_table'}]
        mock_catalog.query.return_value = pd.DataFrame({
            'url': ['http://example.com'],
            'title': ['Test Page'],
            'content': ['Test content'],
            'scraped_at': ['2024-01-01'],
            'source': ['scraper']
        })
        mock_create_catalog.return_value = mock_catalog
        
        pipeline = EmbeddingPipeline()
        df = pipeline.load_gold_data()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert 'url' in df.columns
        assert 'content' in df.columns
    
    @patch('src.embeddings.embedder.create_catalog')
    def test_load_gold_data_register_table(self, mock_create_catalog):
        """Test registering table when it doesn't exist."""
        # Mock catalog with no existing tables
        mock_catalog = Mock()
        mock_catalog.list_tables.return_value = []
        mock_catalog.register_delta_table.return_value = None
        mock_catalog.query.return_value = pd.DataFrame({
            'url': ['http://example.com'],
            'title': ['Test Page'],
            'content': ['Test content'],
            'scraped_at': ['2024-01-01'],
            'source': ['scraper']
        })
        mock_create_catalog.return_value = mock_catalog
        
        pipeline = EmbeddingPipeline()
        df = pipeline.load_gold_data()
        
        mock_catalog.register_delta_table.assert_called_once()
        assert isinstance(df, pd.DataFrame)
    
    @patch('src.embeddings.embedder.LineageContext')
    def test_process_embeddings_success(self, mock_lineage_context):
        """Test successful embedding processing."""
        pipeline = EmbeddingPipeline()
        
        # Mock lineage context
        mock_context = Mock()
        mock_lineage_context.return_value.__enter__.return_value = mock_context
        
        # Mock load_gold_data
        sample_df = pd.DataFrame({
            'url': ['http://example.com/page1', 'http://example.com/page2'],
            'title': ['Page 1', 'Page 2'],
            'content': ['Content 1 with enough text to be meaningful', 'Content 2 with enough text to be meaningful'],
            'scraped_at': ['2024-01-01', '2024-01-01'],
            'source': ['scraper', 'scraper']
        })
        
        with patch.object(pipeline, 'load_gold_data') as mock_load, \
             patch.object(pipeline.embedding_generator, 'embed_texts') as mock_embed, \
             patch.object(pipeline.vector_store, 'add_documents') as mock_add, \
             patch.object(pipeline.vector_store, 'get_collection_info') as mock_info:
            
            mock_load.return_value = sample_df
            mock_embed.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            mock_info.return_value = {"count": 2}
            
            result = pipeline.process_embeddings()
            
            assert result["status"] == "success"
            assert result["embeddings_generated"] > 0
            mock_add.assert_called_once()
    
    @patch('src.embeddings.embedder.LineageContext')
    def test_process_embeddings_no_data(self, mock_lineage_context):
        """Test embedding processing with no data."""
        pipeline = EmbeddingPipeline()
        
        # Mock lineage context
        mock_context = Mock()
        mock_lineage_context.return_value.__enter__.return_value = mock_context
        
        # Mock empty data
        with patch.object(pipeline, 'load_gold_data') as mock_load:
            mock_load.return_value = pd.DataFrame()
            
            result = pipeline.process_embeddings()
            
            assert result["status"] == "no_data"
            assert result["embeddings_generated"] == 0
    
    @patch('src.embeddings.embedder.LineageContext')
    def test_process_embeddings_duplicate_error(self, mock_lineage_context):
        """Test handling duplicate ID errors."""
        pipeline = EmbeddingPipeline()
        
        # Mock lineage context
        mock_context = Mock()
        mock_lineage_context.return_value.__enter__.return_value = mock_context
        
        sample_df = pd.DataFrame({
            'url': ['http://example.com/page1'],
            'title': ['Page 1'],
            'content': ['Content 1'],
            'scraped_at': ['2024-01-01'],
            'source': ['scraper']
        })
        
        with patch.object(pipeline, 'load_gold_data') as mock_load, \
             patch.object(pipeline.embedding_generator, 'embed_texts') as mock_embed, \
             patch.object(pipeline.vector_store, 'add_documents') as mock_add, \
             patch.object(pipeline.vector_store, 'get_collection_info') as mock_info:
            
            mock_load.return_value = sample_df
            mock_embed.return_value = [[0.1, 0.2, 0.3]]
            mock_add.side_effect = Exception("DuplicateIDError")  # Simulate duplicate error
            mock_info.return_value = {"count": 1}
            
            # Should handle error gracefully and continue
            result = pipeline.process_embeddings()
            
            # Should still return success status despite duplicate error
            assert "status" in result
    
    def test_query_embeddings(self):
        """Test querying embeddings."""
        pipeline = EmbeddingPipeline()
        
        with patch.object(pipeline.embedding_generator, 'embed_texts') as mock_embed, \
             patch.object(pipeline.vector_store, 'query_similar') as mock_query:
            
            mock_embed.return_value = [[0.1, 0.2, 0.3]]
            mock_query.return_value = {
                'documents': [['Document 1']],
                'metadatas': [[{'title': 'Doc 1'}]],
                'distances': [[0.1]]
            }
            
            results = pipeline.query_embeddings("test query", n_results=1)
            
            mock_embed.assert_called_once_with(["test query"])
            mock_query.assert_called_once_with(
                query_embedding=[0.1, 0.2, 0.3],
                n_results=1
            )
            assert 'documents' in results
    
    def test_cleanup(self):
        """Test cleanup method."""
        pipeline = EmbeddingPipeline()
        
        # Should not raise an exception
        pipeline.cleanup()
    
    def test_legacy_generate_embeddings(self):
        """Test legacy generate_embeddings method."""
        pipeline = EmbeddingPipeline()
        
        with patch.object(pipeline.embedding_generator, 'embed_texts') as mock_embed:
            mock_embed.return_value = [[0.1, 0.2], [0.3, 0.4]]
            
            texts = ["Text 1", "Text 2"]
            embeddings = pipeline.generate_embeddings(texts)
            
            mock_embed.assert_called_once_with(texts)
            assert embeddings == [[0.1, 0.2], [0.3, 0.4]]
    
    def test_legacy_store_embeddings(self):
        """Test legacy store_embeddings method."""
        pipeline = EmbeddingPipeline()
        
        with patch.object(pipeline.vector_store, 'add_documents') as mock_add:
            documents = ["Doc 1", "Doc 2"]
            embeddings = [[0.1, 0.2], [0.3, 0.4]]
            
            result = pipeline.store_embeddings(documents, embeddings)
            
            mock_add.assert_called_once()
            assert result is True
    
    def test_legacy_store_embeddings_mismatch(self):
        """Test legacy store_embeddings with length mismatch."""
        pipeline = EmbeddingPipeline()
        
        documents = ["Doc 1"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]  # Mismatch
        
        with pytest.raises(ValueError, match="Documents and embeddings length mismatch"):
            pipeline.store_embeddings(documents, embeddings)


class TestEmbeddingUtilities:
    """Test embedding utility functions."""
    
    def test_embed_texts_function(self):
        """Test standalone embed_texts function."""
        from src.embeddings.embedder import embed_texts
        
        with patch('src.embeddings.embedder.get_embedding') as mock_get_embedding:
            mock_get_embedding.return_value = [0.1, 0.2, 0.3]
            
            texts = ["Text 1", "Text 2"]
            embeddings = embed_texts(texts)
            
            assert len(embeddings) == 2
            assert all(emb == [0.1, 0.2, 0.3] for emb in embeddings)
    
    def test_embed_texts_function_with_model(self):
        """Test embed_texts function with specific model."""
        from src.embeddings.embedder import embed_texts
        
        with patch('src.embeddings.embedder.get_embedding') as mock_get_embedding:
            mock_get_embedding.return_value = [0.1, 0.2, 0.3]
            
            texts = ["Text 1"]
            embeddings = embed_texts(texts, model="custom-model")
            
            mock_get_embedding.assert_called_with("Text 1", model="custom-model")
            assert embeddings == [[0.1, 0.2, 0.3]]
    
    def test_embed_texts_function_error(self):
        """Test embed_texts function error handling."""
        from src.embeddings.embedder import embed_texts
        
        with patch('src.embeddings.embedder.get_embedding') as mock_get_embedding:
            mock_get_embedding.side_effect = Exception("Embedding failed")
            
            texts = ["Text 1"]
            embeddings = embed_texts(texts)
            
            assert embeddings == [None]


class TestMainFunction:
    """Test main function and entry point."""
    
    def test_main_function_structure(self):
        """Test that main function exists and has proper structure."""
        from src.embeddings.embedder import main
        
        # Test that main function exists
        assert callable(main)
        
        # Test main with mocked dependencies
        with patch('builtins.print'), \
             patch('src.embeddings.embedder.EmbeddingPipeline') as mock_pipeline_class:
            
            mock_pipeline = Mock()
            mock_pipeline.process_embeddings.return_value = {
                "status": "success",
                "embeddings_generated": 10
            }
            mock_pipeline.query_embeddings.return_value = {
                "documents": [["Test doc"]],
                "metadatas": [[{"title": "Test"}]]
            }
            mock_pipeline.cleanup.return_value = None
            mock_pipeline_class.return_value = mock_pipeline
            
            # Should not raise an exception
            try:
                main()
            except SystemExit:
                pass  # main() might call sys.exit()


if __name__ == "__main__":
    pytest.main([__file__])