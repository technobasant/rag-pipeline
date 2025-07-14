#!/bin/sh
set -e

# Start Ollama server in the background
ollama serve &
SERVER_PID=$!

# Wait until the server is up
echo "[Ollama Entrypoint] Waiting for Ollama server to be ready..."
until curl -sf http://localhost:11434 >/dev/null 2>&1; do
  sleep 1
done

echo "[Ollama Entrypoint] Pulling models..."
# Pull required models (skip if already present)
ollama pull mahonzhan/all-MiniLM-L6-v2 || true
ollama pull qwen3:0.6b || true

echo "[Ollama Entrypoint] Models pulled. Ollama ready."

# Bring the Ollama process to the foreground so container stays alive
wait $SERVER_PID

