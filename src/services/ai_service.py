# src/services/ai_service.py

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

from src.utils.prettify_model_name import prettify_model_name

# Настройка логгера
logger = logging.getLogger("ai_service")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

load_dotenv()

# ════════════════════════════════════════════════════════════════
# API ключи
# ════════════════════════════════════════════════════════════════
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ════════════════════════════════════════════════════════════════
# Модели для Arena
# Используем Hugging Face как основной бесплатный сервис,
# OpenRouter оставляем как fallback.
# ════════════════════════════════════════════════════════════════
AVAILABLE_MODELS = {
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "zephyr-7b": "HuggingFaceH4/zephyr-7b-beta",
    "mistral-nemo": "mistralai/Mistral-Nemo-Instruct-2407",
    "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct:free",
    "llama-3.3-70b": "meta-llama/llama-3.3-70b-instruct:free",
}

JUDGE_MODEL = {
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "zephyr-7b": "HuggingFaceH4/zephyr-7b-beta",
    "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct:free",
}

DEFAULT_MODELS = ["mistral-7b", "zephyr-7b"]
DEFAULT_JUDGE = "mistral-7b"

MODEL_PROVIDER_BY_ID = {
    "mistralai/Mistral-7B-Instruct-v0.3": "huggingface",
    "HuggingFaceH4/zephyr-7b-beta": "huggingface",
    "mistralai/Mistral-Nemo-Instruct-2407": "huggingface",
    "meta-llama/llama-3.1-8b-instruct:free": "openrouter",
    "meta-llama/llama-3.3-70b-instruct:free": "openrouter",
}


def load_prompt(filename: str) -> str:
    prompt_dir = Path(__file__).resolve().parent.parent / "prompts"
    file_path = prompt_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    return file_path.read_text(encoding="utf-8").strip()


SYSTEM_PROMPT = load_prompt("system_prompt.md")
JUDGE_PROMPT_TEMPLATE = load_prompt("judge_prompt.md")


# ════════════════════════════════════════════════════════════════
# HUGGING FACE INFERENCE API
# ════════════════════════════════════════════════════════════════

async def fetch_from_huggingface(session, model_id, prompt, temperature=0.0, max_tokens=2000):
    if not HUGGINGFACE_API_KEY:
        logger.error("HUGGINGFACE_API_KEY не задан")
        return {"model": model_id, "content": "Ошибка: HUGGINGFACE_API_KEY не задан", "status": "error"}

    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.95,
            "repetition_penalty": 1.05,
            "do_sample": temperature > 0,
        },
        "options": {
            "wait_for_model": True,
        },
    }

    try:
        async with session.post(
            url,
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    content = data[0].get("generated_text", "")
                    if content.startswith(prompt):
                        content = content[len(prompt):].strip()
                    return {"model": model_id, "content": content, "status": "success"}
                if isinstance(data, dict) and "generated_text" in data:
                    content = data["generated_text"]
                    if content.startswith(prompt):
                        content = content[len(prompt):].strip()
                    return {"model": model_id, "content": content, "status": "success"}
                logger.error("Непредсказуемый ответ Hugging Face: %s", data)
                return {"model": model_id, "content": "Hugging Face: неожиданный формат ответа", "status": "error"}

            error_text = await resp.text()
            logger.error("Hugging Face ошибка %s: %s", resp.status, error_text[:250])
            return {"model": model_id, "content": f"Hugging Face ошибка {resp.status}", "status": "error"}

    except asyncio.TimeoutError:
        logger.error("Hugging Face timeout для %s", model_id)
        return {"model": model_id, "content": "Timeout: Hugging Face запрос слишком долго", "status": "error"}
    except Exception as e:
        logger.exception("Исключение Hugging Face для %s", model_id)
        return {"model": model_id, "content": f"Hugging Face исключение: {str(e)}", "status": "error"}


# ════════════════════════════════════════════════════════════════
# OPENROUTER API (fallback)
# ════════════════════════════════════════════════════════════════

async def fetch_from_openrouter(session, model_id, prompt, temperature=0.0, max_tokens=2000):
    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY не задан")
        return {"model": model_id, "content": "Ошибка: OpenRouter API-ключ не задан", "status": "error"}

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/AleKolar/LLM-Judge-for-Arena-battles",
        "User-Agent": "LLMArenaBot/3.0",
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        async with session.post(
            url,
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                content = data["choices"][0]["message"]["content"]
                return {"model": model_id, "content": content, "status": "success"}

            error_text = await resp.text()
            logger.error("OpenRouter ошибка %s: %s", resp.status, error_text[:250])
            return {"model": model_id, "content": f"OpenRouter ошибка {resp.status}", "status": "error"}

    except asyncio.TimeoutError:
        logger.error("OpenRouter timeout для %s", model_id)
        return {"model": model_id, "content": "Timeout: OpenRouter запрос слишком долго", "status": "error"}
    except Exception as e:
        logger.exception("Исключение OpenRouter для %s", model_id)
        return {"model": model_id, "content": f"OpenRouter исключение: {str(e)}", "status": "error"}


# ════════════════════════════════════════════════════════════════
# Универсальный fetch
# ════════════════════════════════════════════════════════════════

async def fetch_from_model(session, model_id, prompt, temperature=0.0, max_tokens=2000):
    provider = MODEL_PROVIDER_BY_ID.get(model_id)
    if provider is None:
        if ":free" in model_id or model_id.startswith("meta-llama/"):
            provider = "openrouter"
        else:
            provider = "huggingface"

    if provider == "huggingface":
        return await fetch_from_huggingface(session, model_id, prompt, temperature, max_tokens)
    return await fetch_from_openrouter(session, model_id, prompt, temperature, max_tokens)


async def compare_models(models, session, custom_prompt=None):
    prompt = custom_prompt or SYSTEM_PROMPT
    selected_ids = [AVAILABLE_MODELS[m] for m in models if m in AVAILABLE_MODELS]
    if not selected_ids:
        logger.warning("Не выбрано ни одной модели из списка %s", models)
        return {"error": "Не выбрано ни одной модели"}

    logger.info("Запуск сравнения моделей: %s", selected_ids)
    tasks = [fetch_from_model(session, mid, prompt) for mid in selected_ids]
    results = await asyncio.gather(*tasks)
    return {"results": results}


def extract_json(content: str) -> dict:
    content = content.strip()
    content = re.sub(r"^```json\s*", "", content, flags=re.IGNORECASE).strip()
    content = re.sub(r"\s*```$", "", content).strip()

    strict = r'\{\s*"winner"\s*:\s*"(MODEL_A|MODEL_B|DRAW)"\s*,\s*"reason"\s*:\s*"([^"]*)"\s*\}'
    m = re.search(strict, content, re.DOTALL)
    if m:
        return {"winner": m.group(1), "reason": m.group(2)}

    loose = r'\{\s*"winner"\s*:\s*(MODEL_A|MODEL_B|DRAW)\s*,\s*"reason"\s*:\s*"([^"]*)"\s*\}'
    m = re.search(loose, content, re.DOTALL)
    if m:
        return {"winner": m.group(1), "reason": m.group(2)}

    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        json_str = content[start:end + 1]
        try:
            obj = json.loads(json_str)
            if "winner" in obj and "reason" in obj:
                return obj
        except json.JSONDecodeError:
            pass

    raise ValueError("JSON объект не найден")


async def ask_judge(session, model1, response1, model2, response2, judge_model_name=DEFAULT_JUDGE):
    safe_resp1 = response1.replace("{", "{{").replace("}", "}}")
    safe_resp2 = response2.replace("{", "{{").replace("}", "}}")
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        model_a_name=model1,
        response_a=safe_resp1,
        model_b_name=model2,
        response_b=safe_resp2,
    )

    if judge_model_name not in JUDGE_MODEL:
        logger.warning("Судья %s не найден, используем default", judge_model_name)
        judge_model_name = DEFAULT_JUDGE

    judge_id = JUDGE_MODEL[judge_model_name]
    logger.info("Отправка запроса судье %s (%s)", judge_model_name, judge_id)
    response = await fetch_from_model(session, judge_id, prompt, temperature=0.0, max_tokens=1500)
    if response["status"] != "success":
        logger.error("Судья %s не ответил: %s", judge_model_name, response["content"][:200])
        return {"error": f"Судья не ответил: {response['content']}"}

    try:
        verdict = extract_json(response["content"])
        logger.info("Судья %s вернул verdict: %s", judge_model_name, verdict.get("winner"))
    except Exception as e:
        logger.warning("Ошибка парсинга JSON от судьи %s: %s", judge_model_name, str(e))
        return {"error": f"Ошибка парсинга Judge JSON: {str(e)}", "raw_response": response["content"]}

    winner = verdict.get("winner")
    reason = verdict.get("reason")
    if winner not in ["MODEL_A", "MODEL_B", "DRAW"]:
        logger.warning("Судья %s вернул неверный winner: %s", judge_model_name, winner)
        return {"error": f"Judge вернул неверный winner: {winner}", "raw_response": response["content"]}
    if not isinstance(reason, str) or not reason.strip():
        logger.warning("Судья %s вернул пустой reason", judge_model_name)
        return {"error": "Judge вернул пустой reason", "raw_response": response["content"]}
    return {"winner": winner, "reason": reason}


async def judge_winner(results, session, judge_model=DEFAULT_JUDGE):
    logger.info("Начало судейства, модель судьи: %s", judge_model)
    successful_results = [r for r in results if r.get("status") == "success"]
    failed_results = [r for r in results if r.get("status") == "error"]

    if len(successful_results) == 0:
        logger.error("Все модели завершились ошибкой")
        return {
            "winners": [],
            "losers": [r["model"] for r in failed_results],
            "message": "❌ Все модели завершились ошибкой.",
            "judge_result": {"winner": None, "reason": "Обе модели не смогли выполнить задание."},
            "reason": "Обе модели не смогли выполнить задание.",
            "evidence": failed_results,
            "winner_position": None,
            "judge_model": judge_model,
        }

    if len(successful_results) == 1:
        winner = successful_results[0]
        loser_model = failed_results[0]["model"] if failed_results else "неизвестная модель"
        winner_pos = "MODEL_A" if winner["model"] == results[0]["model"] else "MODEL_B"
        reason_text = f"Модель {loser_model} завершилась с ошибкой, поэтому побеждает {winner['model']}."
        logger.info("Только одна модель успешна: %s (позиция %s)", winner['model'], winner_pos)
        return {
            "winners": [winner["model"]],
            "losers": [r["model"] for r in failed_results],
            "message": f"🏆 Победитель: {winner['model']}",
            "judge_result": {"winner": winner["model"], "reason": reason_text},
            "reason": reason_text,
            "evidence": results,
            "winner_position": winner_pos,
            "judge_model": judge_model,
        }

    res1, res2 = successful_results[0], successful_results[1]
    model1, model2 = res1["model"], res2["model"]
    response1, response2 = res1["content"], res2["content"]

    if model1 == model2:
        judge_model1 = f"{model1} (MODEL_A)"
        judge_model2 = f"{model2} (MODEL_B)"
    else:
        judge_model1, judge_model2 = model1, model2

    judge_result = await ask_judge(session, judge_model1, response1, judge_model2, response2, judge_model)

    if "error" in judge_result:
        error_detail = judge_result.get("error", "Неизвестная ошибка")
        reason_text = f"Судья не дал корректного вердикта. {error_detail}."
        logger.error("Ошибка судьи %s: %s", judge_model, error_detail)
        return {
            "winners": [],
            "losers": [],
            "message": "❌ Judge не смог определить победителя.",
            "judge_result": {"winner": None, "reason": reason_text},
            "judge_error": judge_result,
            "evidence": results,
            "winner_position": None,
            "judge_model": judge_model,
            "reason": reason_text,
        }

    winner_alias = judge_result["winner"]
    if winner_alias == "DRAW":
        reason = judge_result.get("reason", "")
        logger.info("Судья %s объявил ничью", judge_model)
        return {
            "winners": [],
            "losers": [],
            "message": "🤝 Ничья! Обе модели показали одинаково хороший результат.",
            "judge_result": {"winner": "DRAW", "reason": reason},
            "reason": reason,
            "evidence": results,
            "winner_position": None,
            "judge_model": judge_model,
        }

    winner_position = winner_alias
    winner_map = {"MODEL_A": model1, "MODEL_B": model2}
    winner = winner_map[winner_alias]
    losers = [m for m in [model1, model2] if m != winner]

    reason = judge_result.get("reason", "")
    reason = reason.replace("MODEL_A", judge_model1).replace("MODEL_B", judge_model2)
    judge_result["reason"] = reason

    winner_display = prettify_model_name(winner)
    logger.info("Победитель: %s (позиция %s)", winner, winner_position)

    return {
        "winners": [winner],
        "losers": losers,
        "message": f"🏆 Победитель: {winner_display}",
        "judge_result": judge_result,
        "summary": judge_result.get("summary", "Судья выбрал победителя."),
        "reason": reason,
        "evidence": results,
        "winner_position": winner_position,
        "judge_model": judge_model,
    }


async def run_arena_comparison(models: list[str], session: aiohttp.ClientSession, prompt: str = None) -> dict:
    start = time.time()
    compare_result = await compare_models(models=models, session=session, custom_prompt=prompt)
    elapsed = round(time.time() - start, 2)
    if "error" in compare_result:
        return compare_result
    return {
        "results": compare_result["results"],
        "elapsed": elapsed,
    }
