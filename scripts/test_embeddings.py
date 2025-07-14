from dotenv import load_dotenv
load_dotenv()

import os
import logging
import traceback

required_vars = ["MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY", "MINIO_BUCKET_NAME"]
for var in required_vars:
    if not os.getenv(var):
        print(f"ERROR: {var} is not set. Check your .env or .env.dev file.")

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.embeddings.embedder import EmbeddingPipeline

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        pipeline = EmbeddingPipeline()
        print("Embeddings pipeline initialized successfully.")
        result = pipeline.process_embeddings(reset_collection=True)
        print("\n[Embedding Process Result]")
        print(result)
        if result["status"] == "success":
            query_text = "What is Python?"
            print(f"\n[Querying Embeddings for: '{query_text}']")
            query_result = pipeline.query_embeddings(query_text, n_results=3)
            print("[Query Results]")
            print(query_result)
    except Exception as e:
        print("Embeddings error:", e)
        traceback.print_exc() 