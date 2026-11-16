import asyncio
import json

import aiohttp

MODEL = "hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M"
URL = "http://127.0.0.1:11434/api/chat"


async def main():
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": "Reply with exactly: OLLAMA_LOCAL_OK"
            }
        ],
        "stream": False,
        "keep_alive": "5m",
        "options": {
            "temperature": 0,
            "num_ctx": 4096
        }
    }

    timeout = aiohttp.ClientTimeout(total=180)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(URL, json=payload) as response:
            body = await response.text()

            print("HTTP status:", response.status)

            if response.status != 200:
                print("API error:", body[:2000])
                return

            data = json.loads(body)

            print("Model:", data.get("model"))
            print("Response:", data.get("message", {}).get("content"))
            print("Prompt tokens:", data.get("prompt_eval_count"))
            print("Generated tokens:", data.get("eval_count"))
            print("Thinking:", data.get("message", {}).get("thinking", ""))
            print("Final answer:", data.get("message", {}).get("content"))


if __name__ == "__main__":
    asyncio.run(main())

# python check_ollama.py
