"""
Unit tests for the ETL pipeline module.
Tests are written to match the current implementation in src/etl/bronze_silver_gold.py
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime

# Try to import ETL components
try:
    from src.etl.bronze_silver_gold import (
        SparkETLPipeline, bronze_transformation, silver_transformation, gold_transformation,
        get_spark_session, run_bronze_transformation, run_silver_transformation, 
        run_gold_transformation, register_table_in_duckdb, run_full_pipeline
    )
    ETL_AVAILABLE = True
except ImportError:
    ETL_AVAILABLE = False

# Skip all tests if ETL module is not available
pytestmark = pytest.mark.skipif(not ETL_AVAILABLE, reason="ETL module not available")


class TestSparkSessionCreation:
    """Test Spark session creation and configuration."""
    
    @patch('src.etl.bronze_silver_gold.SparkSession')
    def test_get_spark_session_configuration(self, mock_spark_session):
        """Test Spark session is configured correctly."""
        mock_builder = Mock()
        mock_spark_session.builder = mock_builder
        mock_builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = Mock()
        
        spark = get_spark_session("TestApp")
        
        # Verify builder chain was called
        mock_builder.appName.assert_called_with("TestApp")
        assert mock_builder.config.call_count > 0  # Multiple config calls
        mock_builder.getOrCreate.assert_called_once()
    
    @patch('src.etl.bronze_silver_gold.SparkSession')
    def test_get_spark_session_default_name(self, mock_spark_session):
        """Test default app name."""
        mock_builder = Mock()
        mock_spark_session.builder = mock_builder
        mock_builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = Mock()
        
        spark = get_spark_session()
        
        mock_builder.appName.assert_called_with("ETL-Pipeline")


class TestBronzeTransformation:
    """Test Bronze layer transformation."""
    
    @patch('src.etl.bronze_silver_gold.F')
    def test_bronze_transformation_structure(self, mock_functions):
        """Test Bronze transformation logic structure."""
        mock_spark = Mock()
        
        # Mock DataFrames
        mock_meta_df = Mock()
        mock_text_df = Mock()
        mock_joined_df = Mock()
        mock_bronze_df = Mock()
        
        # Setup DataFrame chain
        mock_spark.read.json.return_value = mock_meta_df
        mock_spark.read.text.return_value = mock_text_df
        mock_text_df.withColumn.return_value = mock_text_df
        mock_meta_df.join.return_value = mock_joined_df
        mock_joined_df.withColumnRenamed.return_value = mock_joined_df
        mock_joined_df.withColumn.return_value = mock_joined_df
        mock_joined_df.select.return_value = mock_bronze_df
        
        # Mock write operations
        mock_write = Mock()
        mock_bronze_df.write = mock_write
        mock_write.mode.return_value = mock_write
        mock_write.parquet.return_value = None
        
        result = bronze_transformation(mock_spark)
        
        # Verify read operations
        mock_spark.read.json.assert_called_once()
        mock_spark.read.text.assert_called_once()
        
        # Verify write operation
        mock_write.mode.assert_called_with("overwrite")
        mock_write.parquet.assert_called_once()
        
        assert result == mock_bronze_df
    
    @patch('src.etl.bronze_silver_gold.F')
    def test_bronze_transformation_column_handling(self, mock_functions):
        """Test Bronze transformation column handling."""
        mock_spark = Mock()
        
        # Mock the column operations
        mock_meta_df = Mock()
        mock_text_df = Mock()
        mock_joined_df = Mock()
        mock_bronze_df = Mock()
        
        # Mock columns attribute
        mock_joined_df.columns = ['url', 'title', 'value', 'timestamp', 'text_key']
        
        # Setup chain
        mock_spark.read.json.return_value = mock_meta_df
        mock_spark.read.text.return_value = mock_text_df
        mock_text_df.withColumn.return_value = mock_text_df
        mock_meta_df.join.return_value = mock_joined_df
        mock_joined_df.withColumnRenamed.return_value = mock_joined_df
        mock_joined_df.withColumn.return_value = mock_joined_df
        mock_joined_df.select.return_value = mock_bronze_df
        
        # Mock write
        mock_bronze_df.write.mode.return_value.parquet.return_value = None
        
        result = bronze_transformation(mock_spark)
        
        # Should handle column renaming and selection
        assert mock_joined_df.withColumnRenamed.call_count >= 1
        assert mock_joined_df.select.call_count == 1


class TestSilverTransformation:
    """Test Silver layer transformation."""
    
    def test_silver_transformation_structure(self):
        """Test Silver transformation logic structure."""
        mock_spark = Mock()
        
        # Mock DataFrames
        mock_bronze_df = Mock()
        mock_silver_df = Mock()
        
        # Setup read operation
        mock_spark.read.parquet.return_value = mock_bronze_df
        mock_bronze_df.createOrReplaceTempView.return_value = None
        
        # Setup SQL operation
        mock_spark.sql.return_value = mock_silver_df
        
        # Mock write operation
        mock_write = Mock()
        mock_silver_df.write = mock_write
        mock_write.mode.return_value = mock_write
        mock_write.parquet.return_value = None
        
        result = silver_transformation(mock_spark)
        
        # Verify operations
        mock_spark.read.parquet.assert_called_once()
        mock_bronze_df.createOrReplaceTempView.assert_called_with("bronze")
        mock_spark.sql.assert_called_once()
        mock_write.mode.assert_called_with("overwrite")
        mock_write.parquet.assert_called_once()
        
        assert result == mock_silver_df
    
    def test_silver_transformation_sql_query(self):
        """Test Silver transformation SQL query structure."""
        mock_spark = Mock()
        mock_bronze_df = Mock()
        mock_silver_df = Mock()
        
        mock_spark.read.parquet.return_value = mock_bronze_df
        mock_bronze_df.createOrReplaceTempView.return_value = None
        mock_spark.sql.return_value = mock_silver_df
        mock_silver_df.write.mode.return_value.parquet.return_value = None
        
        silver_transformation(mock_spark)
        
        # Verify SQL was called with a query
        mock_spark.sql.assert_called_once()
        sql_query = mock_spark.sql.call_args[0][0]
        
        # Check SQL query contains expected elements
        assert "SELECT DISTINCT" in sql_query
        assert "FROM bronze" in sql_query
        assert "WHERE" in sql_query
        assert "url IS NOT NULL" in sql_query
        assert "content IS NOT NULL" in sql_query


class TestGoldTransformation:
    """Test Gold layer transformation."""
    
    def test_gold_transformation_structure(self):
        """Test Gold transformation logic structure."""
        mock_spark = Mock()
        
        # Mock DataFrames
        mock_silver_df = Mock()
        mock_gold_df = Mock()
        
        # Setup operations
        mock_spark.read.parquet.return_value = mock_silver_df
        mock_silver_df.createOrReplaceTempView.return_value = None
        mock_spark.sql.return_value = mock_gold_df
        
        # Mock write operation
        mock_write = Mock()
        mock_gold_df.write = mock_write
        mock_write.mode.return_value = mock_write
        mock_write.parquet.return_value = None
        
        result = gold_transformation(mock_spark)
        
        # Verify operations
        mock_spark.read.parquet.assert_called_once()
        mock_silver_df.createOrReplaceTempView.assert_called_with("silver")
        mock_spark.sql.assert_called_once()
        mock_write.mode.assert_called_with("overwrite")
        mock_write.parquet.assert_called_once()
        
        assert result == mock_gold_df
    
    def test_gold_transformation_sql_query(self):
        """Test Gold transformation SQL query structure."""
        mock_spark = Mock()
        mock_silver_df = Mock()
        mock_gold_df = Mock()
        
        mock_spark.read.parquet.return_value = mock_silver_df
        mock_silver_df.createOrReplaceTempView.return_value = None
        mock_spark.sql.return_value = mock_gold_df
        mock_gold_df.write.mode.return_value.parquet.return_value = None
        
        gold_transformation(mock_spark)
        
        # Verify SQL was called
        mock_spark.sql.assert_called_once()
        sql_query = mock_spark.sql.call_args[0][0]
        
        # Check SQL query contains expected elements
        assert "SELECT" in sql_query
        assert "FROM silver" in sql_query
        assert "LENGTH(content)" in sql_query
        assert "content_length" in sql_query


class TestSparkETLPipeline:
    """Test SparkETLPipeline class."""
    
    @patch('src.etl.bronze_silver_gold.get_spark_session')
    def test_pipeline_init_with_spark(self, mock_get_spark_session):
        """Test pipeline initialization with provided Spark session."""
        mock_spark = Mock()
        
        pipeline = SparkETLPipeline(spark=mock_spark)
        
        assert pipeline.spark == mock_spark
        mock_get_spark_session.assert_not_called()
    
    @patch('src.etl.bronze_silver_gold.get_spark_session')
    def test_pipeline_init_without_spark(self, mock_get_spark_session):
        """Test pipeline initialization without Spark session."""
        mock_spark = Mock()
        mock_get_spark_session.return_value = mock_spark
        
        pipeline = SparkETLPipeline()
        
        assert pipeline.spark == mock_spark
        mock_get_spark_session.assert_called_once()
    
    @patch('src.etl.bronze_silver_gold.bronze_transformation')
    def test_pipeline_bronze_method(self, mock_bronze_transformation):
        """Test pipeline bronze method."""
        mock_spark = Mock()
        mock_result = Mock()
        mock_bronze_transformation.return_value = mock_result
        
        pipeline = SparkETLPipeline(spark=mock_spark)
        result = pipeline.bronze()
        
        mock_bronze_transformation.assert_called_once_with(mock_spark)
        assert result == mock_result
    
    @patch('src.etl.bronze_silver_gold.silver_transformation')
    def test_pipeline_silver_method(self, mock_silver_transformation):
        """Test pipeline silver method."""
        mock_spark = Mock()
        mock_result = Mock()
        mock_silver_transformation.return_value = mock_result
        
        pipeline = SparkETLPipeline(spark=mock_spark)
        result = pipeline.silver()
        
        mock_silver_transformation.assert_called_once_with(mock_spark)
        assert result == mock_result
    
    @patch('src.etl.bronze_silver_gold.gold_transformation')
    def test_pipeline_gold_method(self, mock_gold_transformation):
        """Test pipeline gold method."""
        mock_spark = Mock()
        mock_result = Mock()
        mock_gold_transformation.return_value = mock_result
        
        pipeline = SparkETLPipeline(spark=mock_spark)
        result = pipeline.gold()
        
        mock_gold_transformation.assert_called_once_with(mock_spark)
        assert result == mock_result
    
    def test_pipeline_run_success(self):
        """Test successful pipeline run."""
        mock_spark = Mock()
        pipeline = SparkETLPipeline(spark=mock_spark)
        
        with patch.object(pipeline, 'bronze') as mock_bronze, \
             patch.object(pipeline, 'silver') as mock_silver, \
             patch.object(pipeline, 'gold') as mock_gold:
            
            mock_bronze.return_value = Mock()
            mock_silver.return_value = Mock()
            mock_gold.return_value = Mock()
            
            pipeline.run()
            
            mock_bronze.assert_called_once()
            mock_silver.assert_called_once()
            mock_gold.assert_called_once()
            mock_spark.stop.assert_called_once()
    
    def test_pipeline_run_with_error(self):
        """Test pipeline run with error."""
        mock_spark = Mock()
        pipeline = SparkETLPipeline(spark=mock_spark)
        
        with patch.object(pipeline, 'bronze') as mock_bronze:
            mock_bronze.side_effect = Exception("Bronze transformation failed")
            
            with pytest.raises(Exception, match="Bronze transformation failed"):
                pipeline.run()
            
            # Should still stop Spark session
            mock_spark.stop.assert_called_once()


class TestComputeEngineAlternatives:
    """Test alternative compute engines (Polars, DuckDB)."""
    
    @patch.dict('os.environ', {'COMPUTE_ENGINE': 'polars'})
    @patch('src.etl.bronze_silver_gold.pl', create=True)
    def test_run_bronze_transformation_polars(self, mock_polars):
        """Test Bronze transformation with Polars."""
        mock_df = Mock()
        mock_polars.read_csv.return_value = mock_df
        mock_df.drop_nulls.return_value = mock_df
        mock_df.write_parquet.return_value = None
        
        run_bronze_transformation("input.csv", "output.parquet")
        
        mock_polars.read_csv.assert_called_with("input.csv")
        mock_df.drop_nulls.assert_called_once()
        mock_df.write_parquet.assert_called_with("output.parquet")
    
    @patch.dict('os.environ', {'COMPUTE_ENGINE': 'duckdb'})
    @patch('src.etl.bronze_silver_gold.duckdb', create=True)
    def test_run_bronze_transformation_duckdb(self, mock_duckdb):
        """Test Bronze transformation with DuckDB."""
        mock_con = Mock()
        mock_duckdb.connect.return_value = mock_con
        
        run_bronze_transformation("input.csv", "output.parquet")
        
        mock_duckdb.connect.assert_called_once()
        assert mock_con.execute.call_count >= 2  # CREATE and COPY statements
        mock_con.close.assert_called_once()
    
    @patch.dict('os.environ', {'COMPUTE_ENGINE': 'spark'})
    @patch('src.etl.bronze_silver_gold.SparkSession')
    @patch('src.etl.bronze_silver_gold.register_table_in_duckdb')
    def test_run_bronze_transformation_spark(self, mock_register, mock_spark_session):
        """Test Bronze transformation with Spark (default)."""
        mock_spark = Mock()
        mock_builder = Mock()
        mock_spark_session.builder = mock_builder
        mock_builder.appName.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark
        
        mock_df = Mock()
        mock_spark.read.csv.return_value = mock_df
        mock_df.dropna.return_value = mock_df
        mock_write = Mock()
        mock_df.write = mock_write
        mock_write.format.return_value = mock_write
        mock_write.mode.return_value = mock_write
        mock_write.save.return_value = None
        
        run_bronze_transformation("input.csv", "output.parquet")
        
        mock_spark.read.csv.assert_called_with("input.csv", header=True)
        mock_df.dropna.assert_called_once()
        mock_write.format.assert_called_with("delta")
        mock_write.mode.assert_called_with("overwrite")
        mock_write.save.assert_called_with("output.parquet")
        mock_spark.stop.assert_called_once()
        mock_register.assert_called_once()


class TestDuckDBIntegration:
    """Test DuckDB catalog integration."""
    
    @patch('src.etl.bronze_silver_gold.create_catalog')
    def test_register_table_in_duckdb(self, mock_create_catalog):
        """Test registering table in DuckDB catalog."""
        mock_catalog = Mock()
        mock_create_catalog.return_value = mock_catalog
        
        register_table_in_duckdb("test_table", "s3://bucket/path", "test_schema")
        
        mock_create_catalog.assert_called_with(database_path="data/catalog.duckdb")
        mock_catalog.register_delta_table.assert_called_with(
            "test_table", "s3://bucket/path", schema="test_schema"
        )


class TestMainFunctions:
    """Test main entry point functions."""
    
    @patch('src.etl.bronze_silver_gold.SparkETLPipeline')
    def test_run_full_pipeline(self, mock_pipeline_class):
        """Test run_full_pipeline function."""
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        
        run_full_pipeline()
        
        mock_pipeline_class.assert_called_once()
        mock_pipeline.run.assert_called_once()
    
    @patch('src.etl.bronze_silver_gold.run_full_pipeline')
    def test_main_function(self, mock_run_full_pipeline):
        """Test main function."""
        from src.etl.bronze_silver_gold import main
        
        main()
        
        mock_run_full_pipeline.assert_called_once()


class TestEnvironmentConfiguration:
    """Test environment-based configuration."""
    
    @patch.dict('os.environ', {
        'BRONZE_PATH': 's3a://test/bronze/',
        'SILVER_PATH': 's3a://test/silver/',
        'GOLD_PATH': 's3a://test/gold/',
        'RAW_PATH': 's3a://test/raw/'
    })
    def test_environment_paths(self):
        """Test that environment variables are used for paths."""
        from src.etl.bronze_silver_gold import BRONZE_PATH, SILVER_PATH, GOLD_PATH, RAW_PATH
        
        assert BRONZE_PATH == 's3a://test/bronze/'
        assert SILVER_PATH == 's3a://test/silver/'
        assert GOLD_PATH == 's3a://test/gold/'
        assert RAW_PATH == 's3a://test/raw/'
    
    def test_default_paths(self):
        """Test default paths when environment variables are not set."""
        # Import should work with default values
        from src.etl.bronze_silver_gold import BRONZE_PATH, SILVER_PATH, GOLD_PATH, RAW_PATH
        
        assert isinstance(BRONZE_PATH, str)
        assert isinstance(SILVER_PATH, str)
        assert isinstance(GOLD_PATH, str)
        assert isinstance(RAW_PATH, str)
        
        # Should contain expected path components
        assert "bronze" in BRONZE_PATH.lower()
        assert "silver" in SILVER_PATH.lower()
        assert "gold" in GOLD_PATH.lower()
        assert "raw" in RAW_PATH.lower()


class TestLegacyCompatibility:
    """Test legacy compatibility features."""
    
    def test_configure_spark_with_delta_pip(self):
        """Test legacy configure_spark_with_delta_pip function."""
        from src.etl.bronze_silver_gold import configure_spark_with_delta_pip
        
        # Should be a no-op function
        result = configure_spark_with_delta_pip("arg1", "arg2", key="value")
        assert result is None
    
    def test_lineage_context_import(self):
        """Test LineageContext import/fallback."""
        from src.etl.bronze_silver_gold import LineageContext
        
        # Should be importable (either real or fallback)
        assert LineageContext is not None
        
        # Should be usable as context manager
        with LineageContext() as ctx:
            assert ctx is not None


class TestErrorHandling:
    """Test error handling in ETL operations."""
    
    @patch('src.etl.bronze_silver_gold.SparkSession')
    def test_spark_session_creation_error(self, mock_spark_session):
        """Test handling of Spark session creation errors."""
        mock_spark_session.builder.appName.side_effect = Exception("Spark initialization failed")
        
        with pytest.raises(Exception):
            get_spark_session()
    
    def test_transformation_with_missing_data(self):
        """Test transformations with missing or invalid data."""
        mock_spark = Mock()
        
        # Mock empty DataFrame
        mock_empty_df = Mock()
        mock_spark.read.parquet.return_value = mock_empty_df
        mock_empty_df.createOrReplaceTempView.return_value = None
        mock_spark.sql.return_value = mock_empty_df
        mock_empty_df.write.mode.return_value.parquet.return_value = None
        
        # Should handle empty data gracefully
        result = silver_transformation(mock_spark)
        assert result == mock_empty_df


if __name__ == "__main__":
    pytest.main([__file__])