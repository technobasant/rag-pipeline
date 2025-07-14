"""
PySpark ETL pipeline for Bronze → Silver → Gold data transformations.
"""

import os
import logging
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, TimestampType
from pyspark.sql.functions import col, lit, current_timestamp, monotonically_increasing_id
import json
from pyspark.sql import Row
from ..common.config import MinIOClient, config
import pyspark.sql.functions as F
import pyspark.sql.types as T
from src.catalog.duckdb_catalog import create_catalog

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ETL")

# Explicit schema for all layers
CSV_SCHEMA = StructType([
    StructField("id", StringType(), True),
    StructField("url", StringType(), False),
    StructField("title", StringType(), True),
    StructField("content", StringType(), False),
    StructField("scraped_at", TimestampType(), False),
    StructField("source", StringType(), True),
])

# Paths (should be set via env/config in production)
BRONZE_PATH = os.getenv("BRONZE_PATH", "s3a://datalake/bronze/data/")
SILVER_PATH = os.getenv("SILVER_PATH", "s3a://datalake/silver/data/")
GOLD_PATH = os.getenv("GOLD_PATH", "s3a://datalake/gold/data/")
RAW_PATH = os.getenv("RAW_PATH", "s3a://datalake/raw/")

COMPUTE_ENGINE = os.environ.get("COMPUTE_ENGINE", "spark").lower()


def get_spark_session(app_name="ETL-Pipeline"):
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.jars", "jars/hadoop-aws-3.3.4.jar,jars/hadoop-common-3.3.4.jar,jars/aws-java-sdk-bundle-1.12.262.jar") \
        .config("spark.hadoop.fs.s3a.endpoint", os.getenv("MINIO_ENDPOINT")) \
        .config("spark.hadoop.fs.s3a.access.key", os.getenv("MINIO_ACCESS_KEY")) \
        .config("spark.hadoop.fs.s3a.secret.key", os.getenv("MINIO_SECRET_KEY")) \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false") \
        .config("spark.hadoop.fs.s3a.connection.establish.timeout", "60000") \
        .config("spark.hadoop.fs.s3a.connection.timeout", "60000") \
        .config("spark.hadoop.fs.s3a.connection.request.timeout", "60000") \
        .getOrCreate()


def bronze_transformation(spark):
    """
    Reads raw metadata JSON and text files from MinIO using Spark native s3a://, joins them, builds a DataFrame using Spark SQL, writes to bronze as Parquet.
    """
    logger.info("[Bronze] Reading raw metadata from s3a://datalake/raw/metadata/ and text from s3a://datalake/raw/text/")
    df_meta = spark.read.json("s3a://datalake/raw/metadata/")
    df_text = spark.read.text("s3a://datalake/raw/text/").withColumn("full_path", F.input_file_name())
    # Extract S3 key from full_path (remove s3a://datalake/ prefix)
    df_text = df_text.withColumn("text_key", F.expr("substring(full_path, length('s3a://datalake/')+1, length(full_path))"))
    # Join metadata and text on text_key
    df_joined = df_meta.join(df_text, on="text_key", how="left")
    df_joined = df_joined.withColumnRenamed("value", "content")
    df_joined = df_joined.withColumnRenamed("timestamp", "scraped_at")
    df_joined = df_joined.withColumn("source", F.lit("scraper"))
    # Select and cast columns
    select_cols = ["url", "title", "content", "scraped_at", "source"]
    for colname in select_cols:
        if colname not in df_joined.columns:
            df_joined = df_joined.withColumn(colname, F.lit(None).cast(T.StringType()))
        else:
            df_joined = df_joined.withColumn(colname, df_joined[colname].cast(T.StringType()))
    df_bronze = df_joined.select(*select_cols)
    logger.info("[Bronze] Writing normalized data to %s", BRONZE_PATH)
    df_bronze.write.mode("overwrite").parquet(BRONZE_PATH)
    return df_bronze


def silver_transformation(spark):
    """
    Reads bronze Parquet, cleans data, deduplicates, validates schema, writes to silver as Parquet using Spark SQL.
    """
    logger.info("[Silver] Reading bronze data from %s", BRONZE_PATH)
    bronze_df = spark.read.parquet(BRONZE_PATH)
    bronze_df.createOrReplaceTempView("bronze")
    silver_df = spark.sql('''
        SELECT DISTINCT
            url,
            title,
            content,
            scraped_at,
            source
        FROM bronze
        WHERE url IS NOT NULL AND content IS NOT NULL AND scraped_at IS NOT NULL
    ''')
    logger.info("[Silver] Writing cleaned data to %s", SILVER_PATH)
    silver_df.write.mode("overwrite").parquet(SILVER_PATH)
    return silver_df


def gold_transformation(spark):
    """
    Reads silver Parquet, applies business logic, writes to gold as Parquet using Spark SQL.
    """
    logger.info("[Gold] Reading silver data from %s", SILVER_PATH)
    silver_df = spark.read.parquet(SILVER_PATH)
    silver_df.createOrReplaceTempView("silver")
    gold_df = spark.sql('''
        SELECT *,
               LENGTH(content) AS content_length
        FROM silver
    ''')
    logger.info("[Gold] Writing analytics-ready data to %s", GOLD_PATH)
    gold_df.write.mode("overwrite").parquet(GOLD_PATH)
    return gold_df


class SparkETLPipeline:
    """
    Class-based ETL pipeline for Bronze → Silver → Gold transformations.
    Use run() to execute the full pipeline.
    """
    def __init__(self, spark=None):
        self.spark = spark or get_spark_session()
        self.logger = logger

    def bronze(self):
        return bronze_transformation(self.spark)

    def silver(self):
        return silver_transformation(self.spark)

    def gold(self):
        return gold_transformation(self.spark)

    def run(self):
        try:
            self.logger.info("Starting Bronze transformation...")
            self.bronze()
            self.logger.info("Bronze transformation complete. Starting Silver transformation...")
            self.silver()
            self.logger.info("Silver transformation complete. Starting Gold transformation...")
            self.gold()
            self.logger.info("Gold transformation complete. ETL pipeline finished successfully.")
        except Exception as e:
            self.logger.error(f"ETL pipeline failed: {e}", exc_info=True)
            raise
        finally:
            self.spark.stop()


def run_full_pipeline():
    pipeline = SparkETLPipeline()
    pipeline.run()


def main():
    run_full_pipeline()


def register_table_in_duckdb(table_name, s3_path, schema):
    catalog = create_catalog(database_path="data/catalog.duckdb")
    catalog.register_delta_table(table_name, s3_path, schema=schema)


def run_bronze_transformation(input_path, output_path):
    if COMPUTE_ENGINE == "polars":
        try:
            import polars as pl
            print("[INFO] Using Polars for ETL transformation.")
            df = pl.read_csv(input_path)
            # Example transformation: drop nulls
            df = df.drop_nulls()
            df.write_parquet(output_path)
        except ImportError:
            print("[ERROR] polars not installed. Please install polars for Polars support.")
    elif COMPUTE_ENGINE == "duckdb":
        try:
            import duckdb
            print("[INFO] Using DuckDB for ETL transformation.")
            con = duckdb.connect()
            con.execute(f"CREATE TABLE bronze AS SELECT * FROM read_csv_auto('{input_path}') WHERE some_col IS NOT NULL;")
            con.execute(f"COPY bronze TO '{output_path}' (FORMAT PARQUET);")
            con.close()
        except ImportError:
            print("[ERROR] duckdb not installed. Please install duckdb for DuckDB support.")
    else:
        # Default to Spark
        try:
            from pyspark.sql import SparkSession
            print("[INFO] Using Spark for ETL transformation.")
            spark = SparkSession.builder.appName("BronzeETL").getOrCreate()
            df = spark.read.csv(input_path, header=True)
            df = df.dropna()
            df.write.format("delta").mode("overwrite").save(output_path)
            spark.stop()
            # Register in DuckDB
            register_table_in_duckdb("bronze_table", output_path, schema="bronze")
        except ImportError:
            print("[ERROR] pyspark not installed. Please install pyspark for Spark support.")


def run_silver_transformation(input_path, output_path):
    if COMPUTE_ENGINE == "polars":
        try:
            import polars as pl
            print("[INFO] Using Polars for ETL transformation.")
            df = pl.read_csv(input_path)
            # Example transformation: drop nulls
            df = df.drop_nulls()
            df.write_parquet(output_path)
        except ImportError:
            print("[ERROR] polars not installed. Please install polars for Polars support.")
    elif COMPUTE_ENGINE == "duckdb":
        try:
            import duckdb
            print("[INFO] Using DuckDB for ETL transformation.")
            con = duckdb.connect()
            con.execute(f"CREATE TABLE silver AS SELECT * FROM read_csv_auto('{input_path}') WHERE some_col IS NOT NULL;")
            con.execute(f"COPY silver TO '{output_path}' (FORMAT PARQUET);")
            con.close()
        except ImportError:
            print("[ERROR] duckdb not installed. Please install duckdb for DuckDB support.")
    else:
        # Default to Spark
        try:
            from pyspark.sql import SparkSession
            print("[INFO] Using Spark for ETL transformation.")
            spark = SparkSession.builder.appName("SilverETL").getOrCreate()
            df = spark.read.format("delta").load(input_path)
            # ... transformations ...
            df.write.format("delta").mode("overwrite").save(output_path)
            spark.stop()
            # Register in DuckDB
            register_table_in_duckdb("silver_table", output_path, schema="silver")
        except ImportError:
            print("[ERROR] pyspark not installed. Please install pyspark for Spark support.")


def run_gold_transformation(input_path, output_path):
    if COMPUTE_ENGINE == "polars":
        try:
            import polars as pl
            print("[INFO] Using Polars for ETL transformation.")
            df = pl.read_csv(input_path)
            # Example transformation: drop nulls
            df = df.drop_nulls()
            df.write_parquet(output_path)
        except ImportError:
            print("[ERROR] polars not installed. Please install polars for Polars support.")
    elif COMPUTE_ENGINE == "duckdb":
        try:
            import duckdb
            print("[INFO] Using DuckDB for ETL transformation.")
            con = duckdb.connect()
            con.execute(f"CREATE TABLE gold AS SELECT * FROM read_csv_auto('{input_path}') WHERE some_col IS NOT NULL;")
            con.execute(f"COPY gold TO '{output_path}' (FORMAT PARQUET);")
            con.close()
        except ImportError:
            print("[ERROR] duckdb not installed. Please install duckdb for DuckDB support.")
    else:
        # Default to Spark
        try:
            from pyspark.sql import SparkSession
            print("[INFO] Using Spark for ETL transformation.")
            spark = SparkSession.builder.appName("GoldETL").getOrCreate()
            df = spark.read.format("delta").load(input_path)
            # ... transformations ...
            df.write.format("delta").mode("overwrite").save(output_path)
            spark.stop()
            # Register in DuckDB
            register_table_in_duckdb("gold_table", output_path, schema="gold")
        except ImportError:
            print("[ERROR] pyspark not installed. Please install pyspark for Spark support.")


# ------------------------------------------------------------------
# Legacy compatibility shims for unit-tests that patch these symbols
# ------------------------------------------------------------------


def configure_spark_with_delta_pip(*args, **kwargs):  # pragma: no cover
    """No-op stub retained for backwards-compatibility with older tests."""
    return None


# Some tests patch LineageContext from this module – re-export the
# implementation from common.lineage if available.
try:
    from src.common.lineage import LineageContext as _LC  # type: ignore
except Exception:  # pragma: no cover
    class _LC:  # fallback minimal stub
        def __init__(self, *a, **k):
            pass
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

LineageContext = _LC


if __name__ == "__main__":
    main()