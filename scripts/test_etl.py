import sys
import os

import logging
import ollama

from src.common.llm_client import OLLAMA_EMBED_MODEL
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()

required_vars = ["MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY", "MINIO_BUCKET_NAME"]
for var in required_vars:
    if not os.getenv(var):
        print(f"ERROR: {var} is not set. Check your .env or .env.dev file.")

import subprocess
import re

def check_java_version():
    try:
        output = subprocess.check_output(["java", "-version"], stderr=subprocess.STDOUT)
        version_line = output.decode().split('\n')[0]
        print(version_line)
        match = re.search(r'"(\d+)\.(\d+)', version_line)
        if match:
            major = int(match.group(1))
            if major >= 17:
                return True
        # For Java 17+, version string is just "17..."
        match = re.search(r'"(\d+)', version_line)
        if match and int(match.group(1)) >= 17:
            return True
        print("Java version is less than 17. Please install Java 17+ and set JAVA_HOME.")
        return False
    except Exception as e:
        print("Java not found or not on PATH:", e)
        return False

print("MinIO endpoint:", os.getenv("MINIO_ENDPOINT"))
print("MinIO access key:", os.getenv("MINIO_ACCESS_KEY"))
print("MinIO secret key:", os.getenv("MINIO_SECRET_KEY"))
print("MinIO bucket:", os.getenv("MINIO_BUCKET_NAME"))

from src.etl.bronze_silver_gold import main as etl_main

def get_embedding(text: str, model: str = None):
    model = model or OLLAMA_EMBED_MODEL
    try:
        response = ollama.embed(model=model, input=text)
        # Ollama returns 'embeddings' (plural), which is a list of vectors
        return response["embeddings"][0]  # For a single input
    except Exception as e:
        logging.error(f"Ollama embedding failed: {e}")
        raise

if __name__ == "__main__":
    if not check_java_version():
        print("Please install Java 17+ and set JAVA_HOME.")
        sys.exit(1)
    try:
        etl_main()
        print("ETL ran successfully (check logs for details).")
    except Exception as e:
        print("ETL error:", e) 