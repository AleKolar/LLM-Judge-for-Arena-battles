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
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "deepseek-chat": "deepseek/deepseek-chat",
    "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct",
    "qwen3-coder-480b": "qwen/qwen3-coder-480b-a35b-instruct:free",
    "llama-3.3-70b": "meta-llama/llama-3.3-70b-instruct:free",
    "llama-3.2-3b": "meta-llama/llama-3.2-3b-instruct:free",
}

DEFAULT_MODELS = ["gpt-4o-mini", "deepseek-chat"]
JUDGE_MODEL = "deepseek-chat"


def load_prompt(filename: str) -> str:
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
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        async with session.post(url, headers=headers, json=payload) as resp:
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
    prompt = custom_prompt or SYSTEM_PROMPT
    selected_ids = [
        AVAILABLE_MODELS[m]
        for m in models
        if m in AVAILABLE_MODELS
    ]
    if not selected_ids:
        return {"error": "Не выбрано ни одной модели"}
    tasks = [fetch_from_model(session, model_id, prompt) for model_id in selected_ids]
    results = await asyncio.gather(*tasks)
    return {"results": results}


def extract_json(content: str) -> dict:
    content = content.strip()
    content = re.sub(r"^```json", "", content, flags=re.IGNORECASE).strip()
    content = re.sub(r"```$", "", content).strip()
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
    # Экранируем фигурные скобки в ответах моделей, чтобы не сломать форматирование шаблона
    safe_resp1 = response1.replace("{", "{{").replace("}", "}}")
    safe_resp2 = response2.replace("{", "{{").replace("}", "}}")
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        model_a_name=model1,
        response_a=safe_resp1,
        model_b_name=model2,
        response_b=safe_resp2,
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
    return {"winner": winner, "reason": reason}


async def judge_winner(
    results: list[dict],
    session: aiohttp.ClientSession
) -> dict:
    successful_results = [r for r in results if r.get("status") == "success"]
    failed_results = [r for r in results if r.get("status") == "error"]

    # Обе упали
    if len(successful_results) == 0:
        return {
            "winners": [],
            "losers": [r["model"] for r in failed_results],
            "message": "❌ Все модели завершились ошибкой.",
            "judge_result": None,
            "reason": "Обе модели не смогли выполнить задание.",
            "evidence": failed_results,
        }

    # Только одна успешна
    if len(successful_results) == 1:
        winner = successful_results[0]
        loser_model = failed_results[0]["model"] if failed_results else "неизвестная модель"
        # Определяем позицию победителя: если его model равен первому элементу в results,
        # значит это MODEL_A, иначе MODEL_B
        winner_pos = "MODEL_A" if winner["model"] == results[0]["model"] else "MODEL_B"
        return {
            "winners": [winner["model"]],
            "losers": [r["model"] for r in failed_results],
            "message": f"🏆 Победитель: {winner['model']} (Другая модель завершилась ошибкой)",
            "judge_result": None,
            "reason": f"Модель {loser_model} завершилась с ошибкой, поэтому побеждает {winner['model']} по умолчанию.",
            "evidence": results,
            "winner_position": winner_pos,
        }

    # Две успешные модели
    res1, res2 = successful_results[0], successful_results[1]
    model1 = res1["model"]
    model2 = res2["model"]
    response1 = res1["content"]
    response2 = res2["content"]

    # Если модели одинаковые – даём им уникальные имена для Judge
    if model1 == model2:
        judge_model1 = f"{model1} (MODEL_A)"
        judge_model2 = f"{model2} (MODEL_B)"
    else:
        judge_model1 = model1
        judge_model2 = model2

    judge_result = await ask_judge(
        session=session,
        model1=judge_model1,
        response1=response1,
        model2=judge_model2,
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

    winner_alias = judge_result["winner"]
    winner_position = winner_alias  # "MODEL_A" или "MODEL_B"
    winner_map = {"MODEL_A": model1, "MODEL_B": model2}
    winner = winner_map[winner_alias]
    losers = [m for m in [model1, model2] if m != winner]

    reason = judge_result.get("reason", "")
    reason = reason.replace("MODEL_A", judge_model1).replace("MODEL_B", judge_model2)
    judge_result["reason"] = reason

    winner_display = prettify_model_name(winner)

    return {
        "winners": [winner],
        "losers": losers,
        "message": f"🏆 Победитель: {winner_display}",
        "judge_result": judge_result,
        "summary": judge_result.get("summary", "Судья выбрал победителя."),
        "reason": reason,
        "evidence": results,
        "winner_position": winner_position,
    }


async def run_arena_comparison(
    models: list[str],
    session: aiohttp.ClientSession,
    prompt: str = None
) -> dict:
    """
    Только запуск моделей, без вызова судьи.
    Возвращает результаты генерации и время выполнения.
    """
    start = time.time()
    compare_result = await compare_models(models=models, session=session, custom_prompt=prompt)
    elapsed = round(time.time() - start, 2)
    if "error" in compare_result:
        return compare_result
    return {
        "results": compare_result["results"],
        "elapsed": elapsed,
    }