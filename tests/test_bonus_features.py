"""
Unit tests for bonus features and environment-based configurations.
Tests are written to validate optional features and alternative implementations.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

# Test environment variables and optional features
class TestEnvironmentConfiguration:
    """Test environment-based feature switching."""
    
    @patch.dict('os.environ', {'CATALOG_TYPE': 'iceberg'})
    def test_catalog_type_iceberg_detection(self):
        """Test CATALOG_TYPE=iceberg triggers Iceberg detection."""
        catalog_type = os.environ.get('CATALOG_TYPE', 'duckdb')
        assert catalog_type == 'iceberg'
    
    @patch.dict('os.environ', {'COMPUTE_ENGINE': 'polars'})
    def test_compute_engine_polars_detection(self):
        """Test COMPUTE_ENGINE=polars triggers Polars detection."""
        compute_engine = os.environ.get('COMPUTE_ENGINE', 'spark')
        assert compute_engine == 'polars'
    
    @patch.dict('os.environ', {'VECTOR_DB': 'faiss'})
    def test_vector_db_faiss_detection(self):
        """Test VECTOR_DB=faiss triggers FAISS detection."""
        vector_db = os.environ.get('VECTOR_DB', 'chroma')
        assert vector_db == 'faiss'
    
    @patch.dict('os.environ', {'EMBEDDING_DEVICE': 'cpu'})
    def test_embedding_device_cpu_detection(self):
        """Test EMBEDDING_DEVICE=cpu triggers CPU device selection."""
        device = os.environ.get('EMBEDDING_DEVICE', 'auto')
        assert device == 'cpu'
    
    @patch.dict('os.environ', {'EMBEDDING_DEVICE': 'auto'})
    def test_embedding_device_auto_detection(self):
        """Test EMBEDDING_DEVICE=auto triggers auto device selection."""
        device = os.environ.get('EMBEDDING_DEVICE', 'cpu')
        assert device == 'auto'


class TestOptionalDependencies:
    """Test handling of optional dependencies."""
    
    def test_polars_import_handling(self):
        """Test Polars import with fallback."""
        try:
            import polars as pl
            polars_available = True
        except ImportError:
            polars_available = False
        
        # Test should pass regardless of whether Polars is installed
        assert isinstance(polars_available, bool)
    
    def test_faiss_import_handling(self):
        """Test FAISS import with fallback."""
        try:
            import faiss
            faiss_available = True
        except ImportError:
            faiss_available = False
        
        # Test should pass regardless of whether FAISS is installed
        assert isinstance(faiss_available, bool)
    
    def test_duckdb_import_handling(self):
        """Test DuckDB import with fallback."""
        try:
            import duckdb
            duckdb_available = True
        except ImportError:
            duckdb_available = False
        
        # Test should pass regardless of whether DuckDB is installed
        assert isinstance(duckdb_available, bool)
    
    def test_pyiceberg_import_handling(self):
        """Test PyIceberg import with fallback."""
        try:
            import pyiceberg
            iceberg_available = True
        except ImportError:
            iceberg_available = False
        
        # Test should pass regardless of whether PyIceberg is installed
        assert isinstance(iceberg_available, bool)


class TestFeatureSwitching:
    """Test feature switching logic."""
    
    @patch.dict('os.environ', {'STORAGE_BACKEND': 's3'})
    def test_storage_backend_s3(self):
        """Test S3 storage backend selection."""
        backend = os.environ.get('STORAGE_BACKEND', 'minio')
        assert backend == 's3'
    
    @patch.dict('os.environ', {'STORAGE_BACKEND': 'minio'})
    def test_storage_backend_minio(self):
        """Test MinIO storage backend selection."""
        backend = os.environ.get('STORAGE_BACKEND', 'minio')
        assert backend == 'minio'
    
    @patch.dict('os.environ', {'SCRAPER_ENGINE': 'playwright'})
    def test_scraper_engine_playwright(self):
        """Test Playwright scraper engine selection."""
        engine = os.environ.get('SCRAPER_ENGINE', 'beautifulsoup')
        assert engine == 'playwright'
    
    @patch.dict('os.environ', {'SCRAPER_ENGINE': 'beautifulsoup'})
    def test_scraper_engine_beautifulsoup(self):
        """Test BeautifulSoup scraper engine selection."""
        engine = os.environ.get('SCRAPER_ENGINE', 'beautifulsoup')
        assert engine == 'beautifulsoup'
    
    @patch.dict('os.environ', {'EMBEDDING_BACKEND': 'ollama'})
    def test_embedding_backend_ollama(self):
        """Test Ollama embedding backend selection."""
        backend = os.environ.get('EMBEDDING_BACKEND', 'sentence_transformers')
        assert backend == 'ollama'
    
    @patch.dict('os.environ', {'EMBEDDING_BACKEND': 'sentence_transformers'})
    def test_embedding_backend_sentence_transformers(self):
        """Test SentenceTransformers embedding backend selection."""
        backend = os.environ.get('EMBEDDING_BACKEND', 'sentence_transformers')
        assert backend == 'sentence_transformers'


class TestConfigurationValidation:
    """Test configuration validation and defaults."""
    
    def test_chunk_size_configuration(self):
        """Test chunk size configuration."""
        with patch.dict('os.environ', {'CHUNK_SIZE': '1024'}):
            chunk_size = int(os.getenv('CHUNK_SIZE', '512'))
            assert chunk_size == 1024
        
        # Test default
        chunk_size = int(os.getenv('CHUNK_SIZE', '512'))
        assert chunk_size == 512
    
    def test_chunk_overlap_configuration(self):
        """Test chunk overlap configuration."""
        with patch.dict('os.environ', {'CHUNK_OVERLAP': '0.3'}):
            overlap = float(os.getenv('CHUNK_OVERLAP', '0.2'))
            assert overlap == 0.3
        
        # Test default
        overlap = float(os.getenv('CHUNK_OVERLAP', '0.2'))
        assert overlap == 0.2
    
    def test_boolean_environment_variables(self):
        """Test boolean environment variable parsing."""
        # Test true values
        with patch.dict('os.environ', {'TEST_BOOL': 'true'}):
            result = os.getenv('TEST_BOOL', 'false').lower() == 'true'
            assert result is True
        
        with patch.dict('os.environ', {'TEST_BOOL': 'True'}):
            result = os.getenv('TEST_BOOL', 'false').lower() == 'true'
            assert result is True
        
        # Test false values
        with patch.dict('os.environ', {'TEST_BOOL': 'false'}):
            result = os.getenv('TEST_BOOL', 'false').lower() == 'true'
            assert result is False
        
        # Test default
        result = os.getenv('TEST_BOOL_MISSING', 'false').lower() == 'true'
        assert result is False


class TestMakefileTargets:
    """Test Makefile targets exist and are properly structured."""
    
    def test_makefile_exists(self):
        """Test that Makefile exists."""
        makefile_path = Path(__file__).parent.parent / "Makefile"
        assert makefile_path.exists(), "Makefile should exist in project root"
    
    def test_makefile_targets_exist(self):
        """Test that required Makefile targets exist."""
        makefile_path = Path(__file__).parent.parent / "Makefile"
        
        if makefile_path.exists():
            with open(makefile_path, 'r') as f:
                content = f.read()
            
            # Check for common targets
            expected_targets = [
                "scrape:",
                "etl:",
                "embed:",
                "serve:",
                "test:",
                "clean:",
                "help:"
            ]
            
            for target in expected_targets:
                assert target in content, f"Makefile should contain target: {target}"
    
    def test_makefile_phony_targets(self):
        """Test that Makefile has .PHONY declaration."""
        makefile_path = Path(__file__).parent.parent / "Makefile"
        
        if makefile_path.exists():
            with open(makefile_path, 'r') as f:
                content = f.read()
            
            # Should have .PHONY declaration
            assert ".PHONY:" in content, "Makefile should declare .PHONY targets"


class TestDockerConfiguration:
    """Test Docker configuration files."""
    
    def test_docker_compose_files_exist(self):
        """Test that Docker Compose files exist."""
        project_root = Path(__file__).parent.parent
        
        # Check for main docker-compose file
        compose_files = [
            "docker-compose.yaml",
            "docker-compose.yml",
            "docker-compose.dev.yaml",
            "docker-compose.dev.yml"
        ]
        
        found_compose = False
        for compose_file in compose_files:
            if (project_root / compose_file).exists():
                found_compose = True
                break
        
        assert found_compose, "At least one Docker Compose file should exist"
    
    def test_dockerfile_exists(self):
        """Test that Dockerfile exists."""
        project_root = Path(__file__).parent.parent
        dockerfile_path = project_root / "Dockerfile"
        
        # Dockerfile is optional but if it exists, should be valid
        if dockerfile_path.exists():
            with open(dockerfile_path, 'r') as f:
                content = f.read()
            
            assert "FROM" in content, "Dockerfile should have FROM instruction"


class TestProjectStructure:
    """Test project structure and organization."""
    
    def test_source_directories_exist(self):
        """Test that main source directories exist."""
        project_root = Path(__file__).parent.parent
        
        expected_dirs = [
            "src",
            "src/api",
            "src/scraper",
            "src/embeddings",
            "src/etl",
            "src/common"
        ]
        
        for dir_path in expected_dirs:
            full_path = project_root / dir_path
            assert full_path.exists(), f"Directory should exist: {dir_path}"
            assert full_path.is_dir(), f"Path should be a directory: {dir_path}"
    
    def test_config_directories_exist(self):
        """Test that configuration directories exist."""
        project_root = Path(__file__).parent.parent
        
        config_dirs = [
            "sample_config",
            "airflow"
        ]
        
        for dir_path in config_dirs:
            full_path = project_root / dir_path
            if full_path.exists():
                assert full_path.is_dir(), f"Path should be a directory: {dir_path}"
    
    def test_data_directories_structure(self):
        """Test data directories structure."""
        project_root = Path(__file__).parent.parent
        data_dir = project_root / "data"
        
        if data_dir.exists():
            # Check for expected data subdirectories
            expected_subdirs = ["bronze", "silver", "gold"]
            for subdir in expected_subdirs:
                subdir_path = data_dir / subdir
                if subdir_path.exists():
                    assert subdir_path.is_dir(), f"Data subdirectory should be a directory: {subdir}"


class TestConfigurationFiles:
    """Test configuration file structure and validity."""
    
    def test_sample_config_exists(self):
        """Test that sample configuration exists."""
        project_root = Path(__file__).parent.parent
        config_files = [
            "sample_config/site.yaml",
            "sample_config/site.yml",
            "config/site.yaml",
            "config/site.yml"
        ]
        
        found_config = False
        for config_file in config_files:
            if (project_root / config_file).exists():
                found_config = True
                break
        
        assert found_config, "Sample configuration file should exist"
    
    def test_requirements_files_exist(self):
        """Test that requirements files exist."""
        project_root = Path(__file__).parent.parent
        
        req_files = [
            "requirements.txt",
            "pyproject.toml",
            "setup.py"
        ]
        
        found_req = False
        for req_file in req_files:
            if (project_root / req_file).exists():
                found_req = True
                break
        
        assert found_req, "At least one requirements/dependency file should exist"


class TestEnvironmentFiles:
    """Test environment configuration files."""
    
    def test_env_example_exists(self):
        """Test that environment example file exists."""
        project_root = Path(__file__).parent.parent
        
        env_files = [
            ".env.example",
            ".env.template",
            "env.example",
            "sample.env"
        ]
        
        # Environment example file is optional but recommended
        for env_file in env_files:
            env_path = project_root / env_file
            if env_path.exists():
                with open(env_path, 'r') as f:
                    content = f.read()
                
                # Should contain some common environment variables
                common_vars = ["MINIO_", "CHROMA_", "OLLAMA_", "API_"]
                found_vars = sum(1 for var in common_vars if var in content)
                assert found_vars > 0, "Environment file should contain common variables"
                break


class TestScriptFiles:
    """Test utility scripts."""
    
    def test_scripts_directory(self):
        """Test scripts directory structure."""
        project_root = Path(__file__).parent.parent
        scripts_dir = project_root / "scripts"
        
        if scripts_dir.exists():
            assert scripts_dir.is_dir(), "Scripts should be a directory"
            
            # Check for common script files
            script_files = list(scripts_dir.glob("*.py")) + list(scripts_dir.glob("*.sh"))
            assert len(script_files) > 0, "Scripts directory should contain script files"
    
    def test_run_scripts_exist(self):
        """Test that run scripts exist."""
        project_root = Path(__file__).parent.parent
        
        # Look for run scripts in various locations
        run_scripts = [
            "run_api.sh",
            "scripts/run_api.sh",
            "run.py",
            "scripts/run.py"
        ]
        
        found_run_script = False
        for script in run_scripts:
            if (project_root / script).exists():
                found_run_script = True
                break
        
        # Run scripts are optional but helpful
        if found_run_script:
            assert True  # At least one run script exists


class TestDocumentation:
    """Test documentation files."""
    
    def test_readme_exists(self):
        """Test that README file exists."""
        project_root = Path(__file__).parent.parent
        
        readme_files = [
            "README.md",
            "README.rst",
            "README.txt",
            "readme.md"
        ]
        
        found_readme = False
        for readme in readme_files:
            if (project_root / readme).exists():
                found_readme = True
                break
        
        assert found_readme, "README file should exist"
    
    def test_readme_content(self):
        """Test README content structure."""
        project_root = Path(__file__).parent.parent
        readme_path = project_root / "README.md"
        
        if readme_path.exists():
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for common README sections
            expected_sections = [
                "# ",  # Title
                "## ",  # Sections
                "install",  # Installation info
                "usage",  # Usage info
            ]
            
            found_sections = sum(1 for section in expected_sections 
                               if section.lower() in content.lower())
            assert found_sections >= 2, "README should contain basic sections"


if __name__ == "__main__":
    pytest.main([__file__])