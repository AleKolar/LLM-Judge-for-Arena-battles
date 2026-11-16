"""Smoke test for DeepEval's OllamaModel integration with a local Qwen3 model."""

from __future__ import annotations

import sys
from typing import Any

from deepeval.models import OllamaModel

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M"


def main() -> int:
    print("Creating DeepEval OllamaModel...")
    model = OllamaModel(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        generation_kwargs={
            "num_ctx": 4096,
            "num_predict": 128,
        },
    )

    print(f"Model: {OLLAMA_MODEL}")
    print(f"Ollama URL: {OLLAMA_BASE_URL}")
    print("Sending test prompt through DeepEval...")

    try:
        result: Any = model.generate(
            'Reply with exactly this text and nothing else: DEEPEVAL_OLLAMA_OK'
        )
    except Exception as exc:
        print("\nDeepEval -> Ollama request FAILED")
        print(f"Exception: {type(exc).__name__}: {exc}")
        return 1

    # DeepEval model adapters may return generated text together with cost metadata.
    generated_text = result[0] if isinstance(result, tuple) else result

    print("\nDeepEval -> Ollama request SUCCEEDED")
    print(f"Return type: {type(result).__name__}")
    print("Generated text:")
    print(generated_text)

    if not isinstance(generated_text, str) or not generated_text.strip():
        print("\nThe request returned no usable text.")
        return 2

    print("\nSmoke test passed: DeepEval successfully generated text via Ollama.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# python check_deepeval_ollama.py
