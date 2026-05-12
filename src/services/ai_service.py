# src/services/ai_service.py

import asyncio
import json
import os
import re
import time
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

from src.utils.prettify_model_name import prettify_model_name

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

AVAILABLE_MODELS = {
    # Старые (проверенные)
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "deepseek-chat": "deepseek/deepseek-chat",
    "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct",

    # Новые (сильные)
    "qwen3-coder-480b": "qwen/qwen3-coder-480b-a35b-instruct:free",
    "llama-3.3-70b": "meta-llama/llama-3.3-70b-instruct:free",

    # Средние
    "llama-3.2-3b": "meta-llama/llama-3.2-3b-instruct:free",
}

DEFAULT_MODELS = ["gpt-4o-mini", "deepseek-chat"]

JUDGE_MODEL = "deepseek-chat"


def load_prompt(filename: str) -> str:
    """Читает prompt из папки prompts."""
    prompt_dir = Path(__file__).resolve().parent.parent / "prompts"
    file_path = prompt_dir / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {file_path}")

    return file_path.read_text(encoding="utf-8").strip()


SYSTEM_PROMPT = load_prompt("system_prompt.md")
JUDGE_PROMPT_TEMPLATE = load_prompt("judge_prompt.md")


async def fetch_from_model(
    session: aiohttp.ClientSession,
    model_id: str,
    prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 1200,
) -> dict:
    """
    Отправляет prompt модели через OpenRouter.
    """

    if not API_KEY:
        return {
            "model": model_id,
            "content": "Ошибка: API-ключ не задан",
            "status": "error"
        }

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        async with session.post(
            url,
            headers=headers,
            json=payload
        ) as resp:

            if resp.status == 200:
                data = await resp.json()

                return {
                    "model": model_id,
                    "content": data["choices"][0]["message"]["content"],
                    "status": "success",
                }

            error_text = await resp.text()

            return {
                "model": model_id,
                "content": f"Ошибка {resp.status}: {error_text[:500]}",
                "status": "error",
            }

    except Exception as e:
        return {
            "model": model_id,
            "content": f"Исключение: {str(e)}",
            "status": "error",
        }


async def compare_models(
    models: list[str],
    session: aiohttp.ClientSession,
    custom_prompt: str = None
) -> dict:
    """
    Запускает выбранные модели параллельно.
    """

    prompt = custom_prompt or SYSTEM_PROMPT

    selected_ids = [
        AVAILABLE_MODELS[m]
        for m in models
        if m in AVAILABLE_MODELS
    ]

    if not selected_ids:
        return {
            "error": "Не выбрано ни одной модели"
        }

    tasks = [
        fetch_from_model(session, model_id, prompt)
        for model_id in selected_ids
    ]

    results = await asyncio.gather(*tasks)

    return {
        "results": results
    }


def extract_json(content: str) -> dict:
    """
    Безопасно извлекает JSON из ответа LLM.
    Поддерживает markdown fences.
    """

    content = content.strip()

    # Удаляем ```json
    content = re.sub(r"^```json", "", content, flags=re.IGNORECASE).strip()

    # Удаляем ```
    content = re.sub(r"```$", "", content).strip()

    # Ищем JSON объект
    start = content.find("{")
    end = content.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("JSON объект не найден")

    json_str = content[start:end + 1]

    return json.loads(json_str)


async def ask_judge(
    session: aiohttp.ClientSession,
    model1: str,
    response1: str,
    model2: str,
    response2: str,
) -> dict:

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        model_a_name=model1,
        response_a=response1,
        model_b_name=model2,
        response_b=response2,
    )

    judge_id = AVAILABLE_MODELS[JUDGE_MODEL]

    response = await fetch_from_model(
        session=session,
        model_id=judge_id,
        prompt=prompt,
        temperature=0.0,
        max_tokens=1500,
    )

    if response["status"] != "success":
        return {"error": f"Судья не ответил: {response['content']}"}

    try:
        verdict = extract_json(response["content"])
    except Exception as e:
        return {
            "error": f"Ошибка парсинга Judge JSON: {str(e)}",
            "raw_response": response["content"]
        }

    winner = verdict.get("winner")
    reason = verdict.get("reason")

    if winner not in ["MODEL_A", "MODEL_B"]:
        return {
            "error": f"Judge вернул неверный winner: {winner}",
            "raw_response": response["content"]
        }

    if not isinstance(reason, str):
        return {
            "error": "Judge не вернул корректный reason",
            "raw_response": response["content"]
        }

    return {
        "winner": winner,
        "reason": reason,
    }


async def judge_winner(
    results: list[dict],
    session: aiohttp.ClientSession
) -> dict:
    """
    Главная judge-логика.
    Только LLM Judge.
    Никаких regex / эвристик / fallback.
    """

    successful_results = [
        r for r in results
        if r.get("status") == "success"
    ]

    failed_results = [
        r for r in results
        if r.get("status") == "error"
    ]

    # Если обе модели упали
    if len(successful_results) == 0:
        return {
            "winners": [],
            "losers": [r["model"] for r in failed_results],
            "message": "❌ Все модели завершились ошибкой.",
            "judge_result": None,
            "reason": "Обе модели не смогли выполнить задание.",
            "evidence": failed_results,
        }

    # Если одна модель упала
    if len(successful_results) == 1:
        winner = successful_results[0]
        loser_model = failed_results[0]["model"] if failed_results else "неизвестная модель"
        return {
            "winners": [winner["model"]],
            "losers": [r["model"] for r in failed_results],
            "message": f"🏆 Победитель: {winner['model']} (вторая модель завершилась ошибкой)",
            "judge_result": None,
            "reason": f"Модель {loser_model} завершилась с ошибкой, поэтому побеждает {winner['model']} по умолчанию.",
            "evidence": results,
        }

    # Две успешные модели
    res1, res2 = successful_results

    model1 = res1["model"]
    model2 = res2["model"]

    response1 = res1["content"]
    response2 = res2["content"]

    judge_result = await ask_judge(
        session=session,
        model1=model1,
        response1=response1,
        model2=model2,
        response2=response2,
    )

    if "error" in judge_result:
        return {
            "winners": [],
            "losers": [],
            "message": "❌ Judge не смог определить победителя.",
            "judge_error": judge_result,
            "evidence": results,
        }

    # winner = judge_result["winner"]
    winner_alias = judge_result["winner"]

    winner_map = {
        "MODEL_A": model1,
        "MODEL_B": model2,
    }

    winner = winner_map[winner_alias]

    # losers = [
    #     model1 if winner != model1 else model2
    # ]

    losers = [
        m for m in [model1, model2]
        if m != winner
    ]

    summary = judge_result.get(
        "summary",
        "Судья выбрал победителя."
    )

    reason = judge_result.get("reason", "")
    reason = reason.replace("MODEL_A", model1).replace("MODEL_B", model2)
    judge_result["reason"] = reason  # обновляем внутри judge_result

    return {
        "winners": [winner],
        "losers": losers,
        "message": f"🏆 Победитель: {prettify_model_name(winner)}",
        "judge_result": judge_result,
        "summary": summary,
        "reason": reason,
        "evidence": results,
    }


async def run_arena_comparison(
    models: list[str],
    session: aiohttp.ClientSession,
    prompt: str = None
) -> dict:
    """
    Полный pipeline:
    1. Запуск моделей
    2. Judge evaluation
    3. Возврат результата
    """

    start = time.time()

    compare_result = await compare_models(
        models=models,
        session=session,
        custom_prompt=prompt
    )

    if "error" in compare_result:
        return compare_result

    judge_result = await judge_winner(
        results=compare_result["results"],
        session=session
    )

    elapsed = round(time.time() - start, 2)

    return {
        "results": compare_result["results"],
        "judge": judge_result,
        "elapsed": elapsed,
    }
