#!/bin/bash
set -e

# Configurable variables
MINIO_ALIAS=myminio
MINIO_URL=http://localhost:9000
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASS=minioadmin
MINIO_BUCKET_NAME=datalake
MINIO_USER=myuser
MINIO_USER_PASS=myuserpassword
MINIO_ACCESS_KEY=myuseraccesskey
MINIO_SECRET_KEY=myusersecretkey
MINIO_POLICY=readwrite-policy

# 1. Set alias
mc alias set "$MINIO_ALIAS" "$MINIO_URL" "$MINIO_ROOT_USER" "$MINIO_ROOT_PASS" || true

# 2. Create user (ignore if exists)
mc admin user add "$MINIO_ALIAS" "$MINIO_USER" "$MINIO_USER_PASS" || true

# 3. Create bucket (ignore if exists)
mc mb "$MINIO_ALIAS/$MINIO_BUCKET_NAME" || true

# 4. Create policy file
cat > /tmp/$MINIO_POLICY.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Action": [
        "s3:GetBucketLocation",
        "s3:ListBucket",
        "s3:ListBucketMultipartUploads"
      ],
      "Effect": "Allow",
      "Resource": ["arn:aws:s3:::$MINIO_BUCKET_NAME"]
    },
    {
      "Action": [
        "s3:AbortMultipartUpload",
        "s3:DeleteObject",
        "s3:GetObject",
        "s3:ListMultipartUploadParts",
        "s3:PutObject"
      ],
      "Effect": "Allow",
      "Resource": ["arn:aws:s3:::$MINIO_BUCKET_NAME/*"]
    }
  ]
}
EOF

# 5. Add and set policy
mc admin policy add "$MINIO_ALIAS" "$MINIO_POLICY" /tmp/$MINIO_POLICY.json || true
mc admin policy set "$MINIO_ALIAS" "$MINIO_POLICY" user="$MINIO_USER"

# 6. Create access key for user
mc admin accesskey create "$MINIO_ALIAS/" "$MINIO_USER" \
  --access-key "$MINIO_ACCESS_KEY" \
  --secret-key "$MINIO_SECRET_KEY" \
  --policy "$MINIO_POLICY" || true

# 7. Print .env lines
cat <<EOL

# Add the following to your .env file:
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=$MINIO_ACCESS_KEY
MINIO_SECRET_KEY=$MINIO_SECRET_KEY
MINIO_BUCKET_NAME=$MINIO_BUCKET_NAME
EOL 