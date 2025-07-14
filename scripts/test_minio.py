import io
from minio import Minio
from minio.error import S3Error

client = Minio(
    "localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False,
)

bucket = "datalake"
try:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
    data = b"Hello, MinIO!"
    client.put_object(
        bucket,
        "hello.txt",
        data=io.BytesIO(data),
        length=len(data),
    )
    print("MinIO upload successful!")
except S3Error as err:
    print("MinIO error:", err) 