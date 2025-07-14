"""
DuckDB catalog integration for querying Delta Lake tables.
Provides fast analytics queries over Delta Lake data using DuckDB.
"""

import os
import json
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import duckdb
import pandas as pd
import structlog

from ..common.config import config
from ..common.lineage import lineage_tracker, LineageContext

CATALOG_TYPE = os.environ.get("CATALOG_TYPE", "duckdb").lower()

if CATALOG_TYPE == "iceberg":
    try:
        from pyiceberg.catalog import load_catalog
        # Example: load Iceberg catalog (assumes local warehouse)
        iceberg_catalog = load_catalog("default", {"type": "rest", "uri": "http://localhost:8181"})
        # List tables (stub)
        print("[INFO] Iceberg tables:", iceberg_catalog.list_tables())
    except ImportError:
        print("[ERROR] pyiceberg not installed. Please install pyiceberg for Iceberg support.")
    except Exception as e:
        print(f"[ERROR] Iceberg catalog error: {e}")
elif CATALOG_TYPE == "hive":
    # Placeholder for Hive-metastore integration
    print("[INFO] Hive-metastore adapter not implemented. Defaulting to DuckDB.")

class DuckDBDeltaCatalog:
    """DuckDB catalog for querying Delta Lake tables with fast analytics."""
    
    def __init__(self, database_path: Optional[str] = None):
        """Initialize DuckDB catalog with Delta Lake support.
        
        Args:
            database_path: Path to DuckDB database file. If None, uses in-memory database.
        """
        self.config = config
        self.logger = structlog.get_logger(__name__)
        
        # Initialize DuckDB connection
        self.database_path = database_path or ":memory:"
        self.conn = duckdb.connect(self.database_path)
        
        # Configure DuckDB for S3/MinIO access
        self._configure_s3_access()
        
        # Install and load required extensions
        self._setup_extensions()
        
        # Initialize catalog metadata
        self.catalog_metadata = {
            "tables": {},
            "schemas": {"bronze": {}, "silver": {}, "gold": {}},
            "last_updated": None
        }
        
        self.logger.info("Initialized DuckDB Delta catalog", 
                        database_path=self.database_path)
    
    def _configure_s3_access(self):
        """Configure DuckDB for S3/MinIO access."""
        try:
            # Configure S3 settings for MinIO
            self.conn.execute(f"""
                SET s3_endpoint='{self.config.minio.endpoint}';
                SET s3_access_key_id='{self.config.minio.access_key}';
                SET s3_secret_access_key='{self.config.minio.secret_key}';
                SET s3_use_ssl=false;
                SET s3_url_style='path';
            """)
            
            self.logger.info("Configured DuckDB S3 access for MinIO")
            
        except Exception as e:
            self.logger.error("Failed to configure S3 access", error=str(e))
            raise
    
    def _setup_extensions(self):
        """Install and load required DuckDB extensions."""
        try:
            # Install required extensions
            extensions = ['httpfs', 'parquet']
            
            for ext in extensions:
                try:
                    self.conn.execute(f"INSTALL {ext};")
                    self.conn.execute(f"LOAD {ext};")
                    self.logger.debug("Loaded DuckDB extension", extension=ext)
                except Exception as e:
                    self.logger.warning("Failed to load extension", 
                                      extension=ext, error=str(e))
            
            self.logger.info("Setup DuckDB extensions")
            
        except Exception as e:
            self.logger.error("Failed to setup extensions", error=str(e))
            raise
    
    def register_delta_table(self, 
                           table_name: str, 
                           delta_path: str, 
                           schema: str = "main") -> bool:
        """Register a Delta Lake table in DuckDB catalog.
        
        Args:
            table_name: Name of the table in catalog
            delta_path: S3 path to Delta Lake table
            schema: Schema name (bronze, silver, gold)
            
        Returns:
            True if registration successful
        """
        try:
            with lineage_tracker.track_operation(
                operation_type="catalog_register",
                inputs=[delta_path],
                outputs=[f"{schema}.{table_name}"]
            ) as ctx:
                
                # For DuckDB, we'll read Delta tables as Parquet files
                # Since DuckDB doesn't natively support Delta format yet,
                # we'll register the underlying Parquet files
                parquet_path = f"{delta_path}/*.parquet"
                
                # Create schema if it doesn't exist
                self.conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")
                
                # Register table as view over Parquet files
                create_view_sql = f"""
                CREATE OR REPLACE VIEW {schema}.{table_name} AS 
                SELECT * FROM read_parquet('{parquet_path}');
                """
                
                self.conn.execute(create_view_sql)
                
                # Update catalog metadata
                if schema not in self.catalog_metadata["schemas"]:
                    self.catalog_metadata["schemas"][schema] = {}
                
                self.catalog_metadata["schemas"][schema][table_name] = {
                    "delta_path": delta_path,
                    "parquet_path": parquet_path,
                    "registered_at": pd.Timestamp.now().isoformat(),
                    "type": "delta_table"
                }
                
                self.catalog_metadata["tables"][f"{schema}.{table_name}"] = {
                    "schema": schema,
                    "name": table_name,
                    "path": delta_path
                }
                
                ctx.add_metadata({
                    "table_name": f"{schema}.{table_name}",
                    "delta_path": delta_path,
                    "registration_method": "parquet_view"
                })
                
                self.logger.info("Registered Delta table in DuckDB catalog",
                               table_name=f"{schema}.{table_name}",
                               delta_path=delta_path)
                
                return True
                
        except Exception as e:
            self.logger.error("Failed to register Delta table",
                            table_name=table_name,
                            delta_path=delta_path,
                            error=str(e))
            return False
    
    def query(self, sql: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame.
        
        Args:
            sql: SQL query string
            
        Returns:
            Query results as pandas DataFrame
        """
        try:
            with lineage_tracker.track_operation(
                operation_type="catalog_query",
                inputs=self._extract_table_references(sql),
                outputs=["query_result"]
            ) as ctx:
                
                result = self.conn.execute(sql).fetchdf()
                
                ctx.add_metadata({
                    "sql": sql,
                    "rows_returned": len(result),
                    "columns": list(result.columns) if not result.empty else []
                })
                
                self.logger.info("Executed DuckDB query",
                               sql=sql[:100] + "..." if len(sql) > 100 else sql,
                               rows_returned=len(result))
                
                return result
                
        except Exception as e:
            self.logger.error("Query execution failed",
                            sql=sql,
                            error=str(e))
            raise
    
    def _extract_table_references(self, sql: str) -> List[str]:
        """Extract table references from SQL query (simple implementation)."""
        # Simple regex-based extraction - could be enhanced with proper SQL parsing
        import re
        
        # Look for FROM and JOIN clauses
        pattern = r'(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_.]*)'
        matches = re.findall(pattern, sql, re.IGNORECASE)
        
        return list(set(matches))
    
    def list_tables(self, schema: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all registered tables in catalog.
        
        Args:
            schema: Optional schema filter
            
        Returns:
            List of table metadata
        """
        try:
            if schema:
                # Get tables for specific schema
                schema_tables = self.catalog_metadata["schemas"].get(schema, {})
                return [
                    {
                        "schema": schema,
                        "name": table_name,
                        **table_info
                    }
                    for table_name, table_info in schema_tables.items()
                ]
            else:
                # Get all tables
                all_tables = []
                for schema_name, schema_tables in self.catalog_metadata["schemas"].items():
                    for table_name, table_info in schema_tables.items():
                        all_tables.append({
                            "schema": schema_name,
                            "name": table_name,
                            **table_info
                        })
                return all_tables
                
        except Exception as e:
            self.logger.error("Failed to list tables", error=str(e))
            return []
    
    def get_table_info(self, table_name: str, schema: str = "main") -> Optional[Dict[str, Any]]:
        """Get detailed information about a table.
        
        Args:
            table_name: Name of the table
            schema: Schema name
            
        Returns:
            Table information or None if not found
        """
        try:
            full_name = f"{schema}.{table_name}"
            
            # Get basic metadata
            table_info = self.catalog_metadata["tables"].get(full_name)
            if not table_info:
                return None
            
            # Get column information
            describe_sql = f"DESCRIBE {full_name};"
            columns_df = self.conn.execute(describe_sql).fetchdf()
            
            # Get row count
            count_sql = f"SELECT COUNT(*) as row_count FROM {full_name};"
            row_count = self.conn.execute(count_sql).fetchone()[0]
            
            return {
                **table_info,
                "columns": columns_df.to_dict('records'),
                "row_count": row_count,
                "full_name": full_name
            }
            
        except Exception as e:
            self.logger.error("Failed to get table info",
                            table_name=table_name,
                            schema=schema,
                            error=str(e))
            return None
    
    def create_analytics_views(self):
        """Create useful analytics views for common queries."""
        try:
            analytics_views = [
                {
                    "name": "content_analytics",
                    "sql": """
                    CREATE OR REPLACE VIEW analytics.content_analytics AS
                    SELECT 
                        g.url,
                        g.title,
                        g.content_length,
                        g.chunk_count,
                        g.processed_at,
                        s.word_count,
                        s.language,
                        s.quality_score
                    FROM gold.processed_content g
                    LEFT JOIN silver.cleaned_content s ON g.url = s.url
                    """
                },
                {
                    "name": "scraping_summary",
                    "sql": """
                    CREATE OR REPLACE VIEW analytics.scraping_summary AS
                    SELECT 
                        DATE(scraped_at) as scrape_date,
                        COUNT(*) as pages_scraped,
                        AVG(content_length) as avg_content_length,
                        SUM(CASE WHEN status_code = 200 THEN 1 ELSE 0 END) as successful_scrapes,
                        AVG(response_time) as avg_response_time
                    FROM bronze.raw_pages
                    GROUP BY DATE(scraped_at)
                    ORDER BY scrape_date DESC
                    """
                }
            ]
            
            # Create analytics schema
            self.conn.execute("CREATE SCHEMA IF NOT EXISTS analytics;")
            
            # Create views
            for view in analytics_views:
                try:
                    self.conn.execute(view["sql"])
                    self.logger.info("Created analytics view", view_name=view["name"])
                except Exception as e:
                    self.logger.warning("Failed to create analytics view",
                                      view_name=view["name"],
                                      error=str(e))
            
            self.logger.info("Created analytics views")
            
        except Exception as e:
            self.logger.error("Failed to create analytics views", error=str(e))
    
    def export_catalog_metadata(self, output_path: str):
        """Export catalog metadata to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        try:
            metadata_with_timestamp = {
                **self.catalog_metadata,
                "exported_at": pd.Timestamp.now().isoformat(),
                "database_path": self.database_path
            }
            
            with open(output_path, 'w') as f:
                json.dump(metadata_with_timestamp, f, indent=2)
            
            self.logger.info("Exported catalog metadata", output_path=output_path)
            
        except Exception as e:
            self.logger.error("Failed to export catalog metadata",
                            output_path=output_path,
                            error=str(e))
    
    def close(self):
        """Close DuckDB connection."""
        try:
            if self.conn:
                self.conn.close()
                self.logger.info("Closed DuckDB connection")
        except Exception as e:
            self.logger.error("Error closing DuckDB connection", error=str(e))
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Convenience function for creating catalog instance
def create_catalog(database_path: Optional[str] = None) -> DuckDBDeltaCatalog:
    """Create a DuckDB Delta catalog instance.
    
    Args:
        database_path: Path to DuckDB database file
        
    Returns:
        DuckDBDeltaCatalog instance
    """
    return DuckDBDeltaCatalog(database_path)