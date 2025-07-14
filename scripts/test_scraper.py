from dotenv import load_dotenv
load_dotenv()

import os

required_vars = ["MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY", "MINIO_BUCKET_NAME"]
for var in required_vars:
    if not os.getenv(var):
        print(f"ERROR: {var} is not set. Check your .env or .env.dev file.")

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.common import config as common_config
from src.scraper.scraper import main as scraper_main

if __name__ == "__main__":
    # Print MinIO config for debugging
    print("MinIO endpoint:", getattr(common_config, 'MINIO_ENDPOINT', "localhost:9000"))
    print("MinIO access key:", getattr(common_config, 'MINIO_ACCESS_KEY', "minioadmin"))
    print("MinIO secret key:", getattr(common_config, 'MINIO_SECRET_KEY', "minioadmin"))
    print("MinIO bucket:", getattr(common_config, 'MINIO_BUCKET_NAME', "datalake"))
    try:
        scraper_main()
        print("Scraper ran successfully (check logs for details).")
    except Exception as e:
        print("Scraper error:", e) 