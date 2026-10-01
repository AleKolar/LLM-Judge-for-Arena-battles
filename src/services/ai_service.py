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
# ДОСТУПНЫЕ МОДЕЛИ И ИХ КОНФИГУРАЦИЯ
# ════════════════════════════════════════════════════════════════

# Hugging Face бесплатные модели (рекомендуется использовать)
# На hf.co можно создать бесплатный Inference Endpoint или использовать публичный API
AVAILABLE_MODELS = {
    "mistral-7b": {
        "provider": "huggingface",
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "display_name": "🟦 Mistral 7B"
    },
    "zephyr-7b": {
        "provider": "huggingface",
        "model_id": "HuggingFaceH4/zephyr-7b-beta",
        "display_name": "⚡ Zephyr 7B"
    },
    "neural-chat-7b": {
        "provider": "huggingface",
        "model_id": "Intel/neural-chat-7b-v3-3",
        "display_name": "🧠 Neural Chat 7B"
    },
    "mistral-nemo": {
        "provider": "huggingface",
        "model_id": "mistralai/Mistral-Nemo-Instruct-2407",
        "display_name": "🚀 Mistral Nemo"
    },
    # OpenRouter как fallback (если HF недоступен)
    "llama-3.1-8b": {
        "provider": "openrouter",
        "model_id": "meta-llama/llama-3.1-8b-instruct:free",
        "display_name": "🦙 Llama 3.1 8B"
    },
    "llama-3.3-70b": {
        "provider": "openrouter",
        "model_id": "meta-llama/llama-3.3-70b-instruct:free",
        "display_name": "🦙 Llama 3.3 70B"
    },
}

# Модели-судьи
JUDGE_MODEL = {
    "mistral-7b": {
        "provider": "huggingface",
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "display_name": "🟦 Mistral 7B (Judge)"
    },
    "zephyr-7b": {
        "provider": "huggingface",
        "model_id": "HuggingFaceH4/zephyr-7b-beta",
        "display_name": "⚡ Zephyr 7B (Judge)"
    },
    "llama-3.1-8b": {
        "provider": "openrouter",
        "model_id": "meta-llama/llama-3.1-8b-instruct:free",
        "display_name": "🦙 Llama 3.1 8B (Judge)"
    },
}

DEFAULT_MODELS = ["mistral-7b", "zephyr-7b"]
DEFAULT_JUDGE = "mistral-7b"


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
    """
    Использует Hugging Face Inference API (бесплатно).
    
    Требует HUGGINGFACE_API_KEY (получить на https://huggingface.co/settings/tokens)
    - Создайте User Access Token с правом "Inference"
    - Добавьте в .env: HUGGINGFACE_API_KEY=hf_xxxxx
    """
    if not HUGGINGFACE_API_KEY:
        logger.error("HUGGINGFACE_API_KEY не задан в .env")
        return {
            "model": model_id, 
            "content": "Ошибка: HUGGINGFACE_API_KEY не задан", 
            "status": "error"
        }

    logger.info("🤗 Hugging Face запрос к %s", model_id)
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
            "wait_for_model": True,  # Дождётся, если модель загружается
        }
    }
    
    try:
        async with session.post(
            url, 
            headers=headers, 
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                # HF возвращает список с generated_text
                if isinstance(data, list) and len(data) > 0:
                    content = data[0].get("generated_text", "")
                    # Удаляем дублирующийся промпт
                    if content.startswith(prompt):
                        content = content[len(prompt):].strip()
                    logger.info("✅ 🤗 Hugging Face успех (%d символов)", len(content))
                    return {"model": model_id, "content": content, "status": "success"}
                else:
                    logger.error("❌ HF неожиданный формат: %s", data)
                    return {
                        "model": model_id, 
                        "content": f"HF: неожиданный формат {type(data)}", 
                        "status": "error"
                    }
            
            error_text = await resp.text()
            logger.error("❌ 🤗 HF ошибка %d: %s", resp.status, error_text[:150])
            return {
                "model": model_id, 
                "content": f"Hugging Face ошибка {resp.status}", 
                "status": "error"
            }
            
    except asyncio.TimeoutError:
        logger.error("⏱️ 🤗 HF TIMEOUT")
        return {
            "model": model_id, 
            "content": "Timeout: Hugging Face запрос слишком долго", 
            "status": "error"
        }
    except Exception as e:
        logger.exception("🤗 HF исключение")
        return {
            "model": model_id, 
            "content": f"HF исключение: {str(e)[:100]}", 
            "status": "error"
        }


# ════════════════════════════════════════════════════════════════
# OPENROUTER API (FALLBACK)
# ════════════════════════════════════════════════════════════════

async def fetch_from_openrouter(session, model_id, prompt, temperature=0.0, max_tokens=2000):
    """
    OpenRouter как резервный провайдер (если HF недоступен).
    """
    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY не задан")
        return {
            "model": model_id, 
            "content": "Ошибка: OpenRouter API-ключ не задан", 
            "status": "error"
        }

    logger.info("🌐 OpenRouter запрос к %s", model_id)
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/AleKolar/LLM-Judge-for-Arena-battles",
        "User-Agent": "LLMArenaBot/3.0"
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
            timeout=aiohttp.ClientTimeout(total=60)
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                content = data["choices"][0]["message"]["content"]
                logger.info("✅ 🌐 OpenRouter успех (%d символов)", len(content))
                return {"model": model_id, "content": content, "status": "success"}
            
            error_text = await resp.text()
            logger.error("❌ 🌐 OpenRouter ошибка %d: %s", resp.status, error_text[:150])
            return {
                "model": model_id, 
                "content": f"OpenRouter ошибка {resp.status}", 
                "status": "error"
            }
            
    except asyncio.TimeoutError:
        logger.error("⏱️ 🌐 OpenRouter TIMEOUT")
        return {
            "model": model_id, 
            "content": "Timeout: OpenRouter запрос слишком долго", 
            "status": "error"
        }
    except Exception as e:
        logger.exception("🌐 OpenRouter исключение")
        return {
            "model": model_id, 
            "content": f"OpenRouter исключение: {str(e)[:100]}", 
            "status": "error"
        }


# ════════════════════════════════════════════════════════════════
# УНИВЕРСАЛЬНЫЙ ФЕТЧЕР
# ════════════════════════════════════════════════════════════════

async def fetch_from_model(session, model_key, prompt, temperature=0.0, max_tokens=2000):
    """
    Универсальный фетчер. Автоматически выбирает провайдера.
    """
    if model_key not in AVAILABLE_MODELS:
        logger.error("Модель %s не найдена", model_key)
        return {
            "model": model_key,
            "content": f"Модель {model_key} не найдена",
            "status": "error"
        }
    
    config = AVAILABLE_MODELS[model_key]
    provider = config["provider"]
    model_id = config["model_id"]
    
    if provider == "huggingface":
        return await fetch_from_huggingface(session, model_id, prompt, temperature, max_tokens)
    elif provider == "openrouter":
        return await fetch_from_openrouter(session, model_id, prompt, temperature, max_tokens)
    else:
        logger.error("Неизвестный провайдер: %s", provider)
        return {
            "model": model_key,
            "content": f"Неизвестный провайдер: {provider}",
            "status": "error"
        }


async def compare_models(models, session, custom_prompt=None):
    """Запускает две модели параллельно."""
    prompt = custom_prompt or SYSTEM_PROMPT
    
    tasks = []
    for model_key in models:
        if model_key not in AVAILABLE_MODELS:
            logger.warning("Модель %s не найдена", model_key)
            continue
        tasks.append(fetch_from_model(session, model_key, prompt))
    
    if not tasks:
        logger.warning("Не выбрано ни одной модели")
        return {"error": "Не выбрано ни одной модели"}
    
    logger.info("📊 Запуск сравнения %d модели(й)", len(tasks))
    results = await asyncio.gather(*tasks)
    return {"results": results}


def extract_json(content: str) -> dict:
    """Парсит JSON из ответа судьи."""
    content = content.strip()
    content = re.sub(r"^```json\s*", "", content, flags=re.IGNORECASE).strip()
    content = re.sub(r"\s*```$", "", content).strip()

    # 1) Строгий шаблон
    strict = r'\{\s*"winner"\s*:\s*"(MODEL_A|MODEL_B|DRAW)"\s*,\s*"reason"\s*:\s*"([^"]*)"\s*\}'
    m = re.search(strict, content, re.DOTALL)
    if m:
        return {"winner": m.group(1), "reason": m.group(2)}

    # 2) Нестрогий шаблон
    loose = r'\{\s*"winner"\s*:\s*(MODEL_A|MODEL_B|DRAW)\s*,\s*"reason"\s*:\s*"([^"]*)"\s*\}'
    m = re.search(loose, content, re.DOTALL)
    if m:
        return {"winner": m.group(1), "reason": m.group(2)}

    # 3) Fallback
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


async def ask_judge(session, model1, response1, model2, response2, judge_model_key=None):
    """Отправляет ответы двух моделей судье."""
    if judge_model_key is None:
        judge_model_key = DEFAULT_JUDGE
    
    if judge_model_key not in JUDGE_MODEL:
        logger.warning("Судья %s не найден, используем default", judge_model_key)
        judge_model_key = DEFAULT_JUDGE
    
    safe_resp1 = response1.replace("{", "{{").replace("}", "}}")
    safe_resp2 = response2.replace("{", "{{").replace("}", "}}")
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        model_a_name=model1,
        response_a=safe_resp1,
        model_b_name=model2,
        response_b=safe_resp2,
    )
    
    logger.info("⚖️ Отправка запроса судье %s", judge_model_key)
    response = await fetch_from_model(session, judge_model_key, prompt, temperature=0.0, max_tokens=1500)
    
    if response["status"] != "success":
        logger.error("❌ Судья не ответил: %s", response["content"][:200])
        return {"error": f"Судья не ответил: {response['content']}"}
    
    try:
        verdict = extract_json(response["content"])
        logger.info("✅ Судья вернул: %s", verdict.get("winner"))
    except Exception as e:
        logger.warning("⚠️ Ошибка парсинга JSON: %s", str(e))
        return {"error": f"Ошибка парсинга JSON: {str(e)}", "raw_response": response["content"]}
    
    winner = verdict.get("winner")
    reason = verdict.get("reason")
    
    if winner not in ["MODEL_A", "MODEL_B", "DRAW"]:
        logger.warning("⚠️ Неверный winner: %s", winner)
        return {"error": f"Неверный winner: {winner}", "raw_response": response["content"]}
    
    if not isinstance(reason, str) or not reason.strip():
        logger.warning("⚠️ Пустой reason")
        return {"error": "Пустой reason", "raw_response": response["content"]}
    
    return {"winner": winner, "reason": reason}


async def judge_winner(results, session, judge_model=None):
    """Определяет победителя на основе ответов моделей."""
    if judge_model is None:
        judge_model = DEFAULT_JUDGE
    
    logger.info("⚖️ Начало судейства, модель: %s", judge_model)
    successful_results = [r for r in results if r.get("status") == "success"]
    failed_results = [r for r in results if r.get("status") == "error"]

    if len(successful_results) == 0:
        logger.error("❌ Все модели завершились ошибкой")
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
        reason_text = f"Модель {loser_model} завершилась с ошибкой, побеждает {winner['model']}."
        logger.info("✅ Одна успешная модель: %s", winner['model'])
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

    # Две успешные модели
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
        logger.error("❌ Ошибка судьи: %s", error_detail)
        return {
            "winners": [],
            "losers": [],
            "message": "❌ Судья не смог определить победителя.",
            "judge_result": {"winner": None, "reason": error_detail},
            "judge_error": judge_result,
            "evidence": results,
            "winner_position": None,
            "judge_model": judge_model,
            "reason": error_detail
        }

    winner_alias = judge_result["winner"]

    if winner_alias == "DRAW":
        reason = judge_result.get("reason", "")
        logger.info("🤝 Ничья!")
        return {
            "winners": [],
            "losers": [],
            "message": "🤝 Ничья!",
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
    logger.info("🏆 Победитель: %s", winner)

    return {
        "winners": [winner],
        "losers": losers,
        "message": f"🏆 Победитель: {winner_display}",
        "judge_result": judge_result,
        "reason": reason,
        "evidence": results,
        "winner_position": winner_position,
        "judge_model": judge_model,
    }


async def run_arena_comparison(
        models: list[str],
        session: aiohttp.ClientSession,
        prompt: str = None,
) -> dict:
    """Запускает модели без судьи. Возвращает результаты и время."""
    start = time.time()
    compare_result = await compare_models(models=models, session=session, custom_prompt=prompt)
    elapsed = round(time.time() - start, 2)
    if "error" in compare_result:
        return compare_result
    return {
        "results": compare_result["results"],
        "elapsed": elapsed,
    }
