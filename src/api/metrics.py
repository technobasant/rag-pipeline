"""
Prometheus metrics for RAG API service.

This module demonstrates how to add monitoring to the API following KISS principles.
"""

from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, REGISTRY
import time
import logging

# Set up logger
logger = logging.getLogger(__name__)

# Use a custom registry to avoid conflicts
_metrics_registry = None

def get_metrics_registry():
    """Get or create metrics registry to avoid duplication."""
    global _metrics_registry
    if _metrics_registry is None:
        _metrics_registry = CollectorRegistry()
    return _metrics_registry

# Initialize metrics with custom registry
def _init_metrics():
    """Initialize metrics with proper registry handling."""
    registry = get_metrics_registry()
    
    # Request metrics
    http_requests_total = Counter(
        'http_requests_total',
        'Total HTTP requests',
        ['method', 'endpoint', 'status'],
        registry=registry
    )

    http_request_duration_seconds = Histogram(
        'http_request_duration_seconds',
        'HTTP request duration in seconds',
        ['method', 'endpoint'],
        registry=registry
    )

    # RAG specific metrics
    rag_queries_total = Counter(
        'rag_queries_total',
        'Total RAG queries processed',
        ['status'],
        registry=registry
    )

    rag_query_duration_seconds = Histogram(
        'rag_query_duration_seconds',
        'RAG query processing time in seconds',
        registry=registry
    )

    rag_context_documents_retrieved = Histogram(
        'rag_context_documents_retrieved',
        'Number of context documents retrieved per query',
        buckets=(0, 1, 2, 3, 5, 10, 20, 50),
        registry=registry
    )

    # Vector database metrics
    vector_search_duration_seconds = Histogram(
        'vector_search_duration_seconds',
        'Time spent searching vector database',
        registry=registry
    )

    # LLM metrics
    llm_requests_total = Counter(
        'llm_requests_total',
        'Total LLM requests',
        ['model', 'status'],
        registry=registry
    )

    llm_response_tokens = Histogram(
        'llm_response_tokens',
        'Number of tokens in LLM response',
        buckets=(0, 50, 100, 200, 500, 1000, 2000, 4000),
        registry=registry
    )

    llm_request_duration_seconds = Histogram(
        'llm_request_duration_seconds',
        'LLM request duration in seconds',
        ['model'],
        registry=registry
    )

    # Service info - use a different approach for service information
    try:
        service_info = Info('rag_api_info', 'RAG API service information', registry=registry)
        # Set the info using the correct method
        service_info.info({
            'version': '1.0.0',
            'model': 'llama3.2:1b'
        })
    except (AttributeError, TypeError) as e:
        # Fallback: use a simple Gauge for service info if Info doesn't work
        logger.warning(f"Info metric failed ({e}), using simple gauge fallback")
        try:
            service_info = Gauge(
                'rag_api_info_gauge',
                'RAG API service information (1=active)',
                ['version', 'model'],
                registry=registry
            )
            service_info.labels(version='1.0.0', model='llama3.2:1b').set(1)
        except (AttributeError, TypeError) as e2:
            # Final fallback: simple gauge without labels
            logger.warning(f"Gauge with labels failed ({e2}), using simple gauge")
            service_info = Gauge(
                'rag_api_info_simple',
                'RAG API service information (1=active)',
                registry=registry
            )
            service_info.set(1)

    # Active connections gauge
    active_connections = Gauge(
        'rag_api_active_connections',
        'Number of active connections',
        registry=registry
    )
    
    return {
        'http_requests_total': http_requests_total,
        'http_request_duration_seconds': http_request_duration_seconds,
        'rag_queries_total': rag_queries_total,
        'rag_query_duration_seconds': rag_query_duration_seconds,
        'rag_context_documents_retrieved': rag_context_documents_retrieved,
        'vector_search_duration_seconds': vector_search_duration_seconds,
        'llm_requests_total': llm_requests_total,
        'llm_response_tokens': llm_response_tokens,
        'llm_request_duration_seconds': llm_request_duration_seconds,
        'service_info': service_info,
        'active_connections': active_connections
    }

# Global metrics instance
_metrics = None

def get_metrics():
    """Get metrics instance, initializing if needed."""
    global _metrics
    if _metrics is None:
        _metrics = _init_metrics()
    return _metrics

# Convenience accessors
def get_http_requests_total():
    return get_metrics()['http_requests_total']

def get_http_request_duration_seconds():
    return get_metrics()['http_request_duration_seconds']

def get_rag_queries_total():
    return get_metrics()['rag_queries_total']

def get_rag_query_duration_seconds():
    return get_metrics()['rag_query_duration_seconds']

def get_rag_context_documents_retrieved():
    return get_metrics()['rag_context_documents_retrieved']

def get_vector_search_duration_seconds():
    return get_metrics()['vector_search_duration_seconds']

def get_llm_requests_total():
    return get_metrics()['llm_requests_total']

def get_llm_response_tokens():
    return get_metrics()['llm_response_tokens']

def get_llm_request_duration_seconds():
    return get_metrics()['llm_request_duration_seconds']

def get_active_connections():
    """Get active connections gauge with error handling."""
    try:
        metrics = get_metrics()
        gauge = metrics['active_connections']
        # Ensure it's a proper Gauge object
        if hasattr(gauge, 'inc') and hasattr(gauge, 'dec'):
            return gauge
        else:
            logger.warning("Active connections metric is not a proper Gauge, creating fallback")
            # Create a simple fallback gauge with a unique name to avoid conflicts
            registry = get_metrics_registry()
            try:
                fallback_gauge = Gauge(
                    'rag_api_active_connections_fallback',
                    'Number of active connections (fallback)',
                    registry=registry
                )
                return fallback_gauge
            except Exception as fallback_error:
                logger.error(f"Failed to create fallback gauge: {fallback_error}")
                # Return a mock object that has inc/dec methods to prevent crashes
                class MockGauge:
                    def inc(self):
                        logger.debug("MockGauge.inc() called")
                    def dec(self):
                        logger.debug("MockGauge.dec() called")
                    def set(self, value):
                        logger.debug(f"MockGauge.set({value}) called")
                return MockGauge()
    except Exception as e:
        logger.error(f"Failed to get active connections metric: {e}")
        # Return a mock object that has inc/dec methods to prevent crashes
        class MockGauge:
            def inc(self):
                logger.debug("MockGauge.inc() called")
            def dec(self):
                logger.debug("MockGauge.dec() called")
            def set(self, value):
                logger.debug(f"MockGauge.set({value}) called")
        return MockGauge()


def track_request_metrics(method: str, endpoint: str, status_code: int, duration: float):
    """Track HTTP request metrics."""
    metrics = get_metrics()
    metrics['http_requests_total'].labels(
        method=method,
        endpoint=endpoint,
        status=str(status_code)
    ).inc()
    
    metrics['http_request_duration_seconds'].labels(
        method=method,
        endpoint=endpoint
    ).observe(duration)


def track_rag_query(success: bool, duration: float, num_documents: int):
    """Track RAG query metrics."""
    metrics = get_metrics()
    status = "success" if success else "error"
    metrics['rag_queries_total'].labels(status=status).inc()
    metrics['rag_query_duration_seconds'].observe(duration)
    metrics['rag_context_documents_retrieved'].observe(num_documents)


def track_vector_search(duration: float):
    """Track vector database search metrics."""
    metrics = get_metrics()
    metrics['vector_search_duration_seconds'].observe(duration)


def track_llm_request(model: str, success: bool, duration: float, tokens: int = 0):
    """Track LLM request metrics."""
    metrics = get_metrics()
    status = "success" if success else "error"
    metrics['llm_requests_total'].labels(model=model, status=status).inc()
    metrics['llm_request_duration_seconds'].labels(model=model).observe(duration)
    if tokens > 0:
        metrics['llm_response_tokens'].observe(tokens)