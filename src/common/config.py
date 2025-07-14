"""
Configuration management and shared utilities for the Data-Scrape-to-RAG pipeline.
"""

import os
import logging
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path

import yaml
import boto3
from minio import Minio
from minio.error import S3Error
import structlog


@dataclass
class StorageConfig:
    """Storage configuration for S3/MinIO switching."""
    backend: str  # "minio" or "s3"
    # MinIO config
    minio_endpoint: Optional[str] = None
    minio_access_key: Optional[str] = None
    minio_secret_key: Optional[str] = None
    minio_bucket_name: Optional[str] = None
    minio_secure: bool = False
    # S3 config
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: Optional[str] = None
    s3_bucket_name: Optional[str] = None
    s3_endpoint_url: Optional[str] = None


@dataclass
class MinIOConfig:
    """MinIO configuration settings."""
    endpoint: str
    access_key: str
    secret_key: str
    bucket_name: str
    secure: bool = False


@dataclass
class ScrapingConfig:
    """Web scraping configuration settings."""
    base_url: str
    max_pages: int
    delay_seconds: float
    user_agent: str
    timeout_seconds: int = 30
    engine: str = "beautifulsoup"  # "beautifulsoup" or "playwright"
    playwright_headless: bool = True
    playwright_timeout: int = 30000


@dataclass
class SparkConfig:
    """Spark configuration settings."""
    master_url: str
    driver_memory: str
    executor_memory: str
    app_name: str = "DataScrapeRAG"


@dataclass
class ChromaConfig:
    """Chroma vector database configuration."""
    host: str
    port: int
    collection_name: str


@dataclass
class OllamaConfig:
    """Ollama LLM configuration."""
    host: str
    port: int
    model: str


@dataclass
class EmbeddingConfig:
    """Embedding configuration."""
    model: str
    device: str  # "cpu" or "cuda"
    batch_size: int


@dataclass
class APIConfig:
    """API configuration."""
    host: str
    port: int
    streaming_enabled: bool
    cors_origins: list


@dataclass
class LineageConfig:
    """Lineage configuration."""
    marquez_enabled: bool
    marquez_url: Optional[str] = None
    marquez_namespace: Optional[str] = None


@dataclass
class MetricsConfig:
    """Metrics configuration."""
    enabled: bool
    port: int


class Config:
    """Central configuration management."""
    
    def __init__(self):
        # Storage configuration with S3/MinIO switching
        self.storage = StorageConfig(
            backend=os.getenv("STORAGE_BACKEND", "minio"),
            # MinIO
            minio_endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
            minio_access_key=os.getenv("MINIO_ROOT_USER", "minioadmin"),
            minio_secret_key=os.getenv("MINIO_ROOT_PASSWORD", "minioadmin123"),
            minio_bucket_name=os.getenv("MINIO_BUCKET_NAME", "datalake"),
            minio_secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
            # S3
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_region=os.getenv("AWS_REGION", "us-east-1"),
            s3_bucket_name=os.getenv("S3_BUCKET_NAME"),
            s3_endpoint_url=os.getenv("S3_ENDPOINT_URL")
        )
        
        # Legacy MinIO config for backward compatibility
        self.minio = MinIOConfig(
            endpoint=self.storage.minio_endpoint,
            access_key=self.storage.minio_access_key,
            secret_key=self.storage.minio_secret_key,
            bucket_name=self.storage.minio_bucket_name,
            secure=self.storage.minio_secure
        )
        
        self.scraping = ScrapingConfig(
            base_url=os.getenv("SCRAPE_BASE_URL", "https://docs.python.org/3/tutorial/"),
            max_pages=int(os.getenv("SCRAPE_MAX_PAGES", "50")),
            delay_seconds=float(os.getenv("SCRAPE_DELAY_SECONDS", "1.0")),
            user_agent=os.getenv("SCRAPE_USER_AGENT", "Mozilla/5.0 (compatible; DataLakeBot/1.0)"),
            timeout_seconds=int(os.getenv("SCRAPE_TIMEOUT_SECONDS", "30")),
            engine=os.getenv("SCRAPER_ENGINE", "beautifulsoup"),
            playwright_headless=os.getenv("PLAYWRIGHT_HEADLESS", "true").lower() == "true",
            playwright_timeout=int(os.getenv("PLAYWRIGHT_TIMEOUT", "30000"))
        )
        
        self.spark = SparkConfig(
            master_url=os.getenv("SPARK_MASTER_URL", "spark://spark-master:7077"),
            driver_memory=os.getenv("SPARK_DRIVER_MEMORY", "1G"),
            executor_memory=os.getenv("SPARK_EXECUTOR_MEMORY", "1G")
        )
        
        self.chroma = ChromaConfig(
            # Use Docker Compose service name and internal port for ChromaDB
            host=os.getenv("CHROMA_HOST", "localhost"),
            port=int(os.getenv("CHROMA_PORT", "8000")),
            collection_name=os.getenv("CHROMA_COLLECTION_NAME", "documents")
        )
        
        self.ollama = OllamaConfig(
            host=os.getenv("OLLAMA_HOST", "localhost"),
            port=int(os.getenv("OLLAMA_PORT", "11434")),
            model=os.getenv("OLLAMA_MODEL", "llama3.2:1b")
        )
        
        self.embedding = EmbeddingConfig(
            model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            device=os.getenv("EMBEDDING_DEVICE", "cpu"),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
        )
        
        self.api = APIConfig(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8001")),
            streaming_enabled=os.getenv("API_STREAMING_ENABLED", "true").lower() == "true",
            cors_origins=os.getenv("API_CORS_ORIGINS", "*").split(",")
        )
        
        self.lineage = LineageConfig(
            marquez_enabled=os.getenv("MARQUEZ_ENABLED", "false").lower() == "true",
            marquez_url=os.getenv("MARQUEZ_URL"),
            marquez_namespace=os.getenv("MARQUEZ_NAMESPACE", "data-scrape-rag")
        )
        
        self.metrics = MetricsConfig(
            enabled=os.getenv("METRICS_ENABLED", "true").lower() == "true",
            port=int(os.getenv("METRICS_PORT", "9090"))
        )
        
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.log_format = os.getenv("LOG_FORMAT", "json")


class StorageClient:
    """Unified storage client for S3 and MinIO."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = structlog.get_logger(__name__)
        
        if config.storage.backend == "s3":
            self.client = self._create_s3_client()
            self.bucket_name = config.storage.s3_bucket_name
        else:  # minio
            self.client = self._create_minio_client()
            self.bucket_name = config.storage.minio_bucket_name
    
    def _create_s3_client(self):
        """Create S3 client."""
        session = boto3.Session(
            aws_access_key_id=self.config.storage.aws_access_key_id,
            aws_secret_access_key=self.config.storage.aws_secret_access_key,
            region_name=self.config.storage.aws_region
        )
        
        kwargs = {}
        if self.config.storage.s3_endpoint_url:
            kwargs['endpoint_url'] = self.config.storage.s3_endpoint_url
        
        return session.client('s3', **kwargs)
    
    def _create_minio_client(self):
        """Create MinIO client."""
        return Minio(
            endpoint=self.config.storage.minio_endpoint,
            access_key=self.config.storage.minio_access_key,
            secret_key=self.config.storage.minio_secret_key,
            secure=self.config.storage.minio_secure
        )
    
    def ensure_bucket_exists(self) -> bool:
        """Ensure the bucket exists, create if it doesn't."""
        try:
            if self.config.storage.backend == "s3":
                try:
                    self.client.head_bucket(Bucket=self.bucket_name)
                except:
                    self.client.create_bucket(Bucket=self.bucket_name)
                    self.logger.info("Created S3 bucket", bucket=self.bucket_name)
            else:  # minio
                if not self.client.bucket_exists(self.bucket_name):
                    self.client.make_bucket(self.bucket_name)
                    self.logger.info("Created MinIO bucket", bucket=self.bucket_name)
            return True
        except Exception as e:
            self.logger.error("Failed to ensure bucket exists", error=str(e))
            return False
    
    def upload_text(self, object_name: str, content: str) -> bool:
        """Upload text content to storage."""
        try:
            if self.config.storage.backend == "s3":
                self.client.put_object(
                    Bucket=self.bucket_name,
                    Key=object_name,
                    Body=content.encode('utf-8'),
                    ContentType='text/plain'
                )
            else:  # minio
                from io import BytesIO
                data = BytesIO(content.encode('utf-8'))
                self.client.put_object(
                    bucket_name=self.bucket_name,
                    object_name=object_name,
                    data=data,
                    length=len(content.encode('utf-8')),
                    content_type='text/plain'
                )
            self.logger.info("Uploaded text object", object_name=object_name)
            return True
        except Exception as e:
            self.logger.error("Failed to upload text", object_name=object_name, error=str(e))
            return False
    
    def download_text(self, object_name: str) -> Optional[str]:
        """Download text content from storage."""
        try:
            if self.config.storage.backend == "s3":
                response = self.client.get_object(Bucket=self.bucket_name, Key=object_name)
                content = response['Body'].read().decode('utf-8')
            else:  # minio
                response = self.client.get_object(self.bucket_name, object_name)
                content = response.read().decode('utf-8')
                response.close()
                response.release_conn()
            return content
        except Exception as e:
            self.logger.error("Failed to download text", object_name=object_name, error=str(e))
            return None
    
    def list_objects(self, prefix: str = "") -> list:
        """List objects with given prefix."""
        try:
            if self.config.storage.backend == "s3":
                response = self.client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=prefix
                )
                return [obj['Key'] for obj in response.get('Contents', [])]
            else:  # minio
                objects = self.client.list_objects(self.bucket_name, prefix=prefix)
                return [obj.object_name for obj in objects]
        except Exception as e:
            self.logger.error("Failed to list objects", prefix=prefix, error=str(e))
            return []


class MinIOClient(StorageClient):
    """MinIO client wrapper for backward compatibility."""
    
    def __init__(self, config: MinIOConfig):
        # Create a full config object with MinIO settings
        full_config = Config()
        super().__init__(full_config)


def setup_logging(log_level: str = "INFO", log_format: str = "json") -> None:
    """Setup structured logging."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    if log_format.lower() == "json":
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    else:
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.dev.ConsoleRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    logging.basicConfig(level=level)


def load_site_config(config_path=None):
    """
    Load the site configuration YAML file. If config_path is not provided, try Airflow and local default locations.
    """
    import os
    import logging
    logger = logging.getLogger(__name__)
    default_paths = [
        '/opt/airflow/sample_config/site.yaml',  # Airflow container
        'sample_config/site.yaml',               # Local run
    ]
    if config_path:
        paths = [config_path] + default_paths
    else:
        paths = default_paths
    for path in paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    import yaml
                    return yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Failed to load site config from {path}: {e}")
                raise
    logger.error(f"Site configuration file not found in any of: {paths}")
    raise FileNotFoundError(f"Site configuration file not found: {paths}")


# Global configuration instance
config = Config()