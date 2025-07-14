import sys
import logging
from src.common.llm_client import get_embedding, generate_completion

logging.basicConfig(level=logging.INFO)


def test_embedding() -> None:
    text = "The quick brown fox jumps over the lazy dog."
    try:
        emb = get_embedding(text)
        print(f"Embedding for: '{text[:30]}...':\n{emb[:8]}... (len={len(emb)})")
    except Exception as e:
        print(f"Embedding test failed: {e}")


def test_completion():
    prompt = "What is the capital of France?"
    try:
        answer = generate_completion(prompt)
        print(f"Completion for: '{prompt}':\n{answer}")
    except Exception as e:
        print(f"Completion test failed: {e}")


if __name__ == "__main__":
    print("Testing LiteLLM Embedding:")
    test_embedding()
    print("\nTesting LiteLLM Completion:")
    test_completion()
