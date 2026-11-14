# src/services/ai_service.py
import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import aiohttp
from dotenv import load_dotenv

from src.utils.prettify_model_name import prettify_model_name

"""LLM Arena service backed by a local Ollama server.

The public result contract is intentionally kept compatible with the existing
application: every model attempt returns {model, content, status}. A failed
model attempt is valid Arena evidence and must not prevent the battle record
from being stored.
"""

# ---------------------------------------------------------------------------
# Logging and project configuration
# ---------------------------------------------------------------------------

logger = logging.getLogger("ai_service")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Ollama configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
).rstrip("/")
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"


def _read_int_env(name: str, default: int, minimum: int = 1) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning("Некорректное значение %s=%r; используется %s", name, raw_value, default)
        return default
    return max(minimum, value)


def _read_float_env(name: str, default: float, minimum: float = 1.0) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError:
        logger.warning("Некорректное значение %s=%r; используется %s", name, raw_value, default)
        return default
    return max(minimum, value)


def _read_bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


OLLAMA_NUM_CTX = _read_int_env("OLLAMA_NUM_CTX", 4096, minimum=512)
OLLAMA_REQUEST_TIMEOUT = _read_float_env("OLLAMA_REQUEST_TIMEOUT", 300.0)
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "0")
OLLAMA_THINK = _read_bool_env("OLLAMA_THINK", True)

# Serialize local inference requests. This is important on the current
# machine: it has 8 GB VRAM, so Arena requests should not load several models
# into GPU memory concurrently.
OLLAMA_REQUEST_SEMAPHORE = asyncio.Semaphore(
    _read_int_env("OLLAMA_MAX_CONCURRENT_REQUESTS", 1, minimum=1)
)

# Exact model tags verified/used with the local Ollama installation.
QWEN3_4B_MODEL_ID = os.getenv(
    "OLLAMA_MODEL_QWEN3_4B",
    "hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M",
)
QWEN3_8B_MODEL_ID = os.getenv(
    "OLLAMA_MODEL_QWEN3_8B",
    "hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M",
)

# ---------------------------------------------------------------------------
# Available Arena participant models
# ---------------------------------------------------------------------------

AVAILABLE_MODELS: dict[str, dict[str, Any]] = {
    "qwen3-4b": {
        "provider": "ollama",
        "model_id": QWEN3_4B_MODEL_ID,
        "display_name": "Qwen3 4B · Q4_K_M · локально",
    },
    "qwen3-8b": {
        "provider": "ollama",
        "model_id": QWEN3_8B_MODEL_ID,
        "display_name": "Qwen3 8B · Q4_K_M · локально",
    },
}

# Compatibility mapping for previously selected model keys. They are not
# exposed by GET /models, but old browser selections and clients can still
# submit them. The display/evidence always identifies the actual Ollama model.
LEGACY_MODEL_ALIASES: dict[str, str] = {
    "llama-3.1-8b": "qwen3-4b",
    "qwen-coder-7b": "qwen3-8b",
    "gemma-3-4b": "qwen3-4b",
    "qwen3-14b": "qwen3-8b",
    "gpt-oss-20b": "qwen3-8b",
    "phi-4": "qwen3-4b",
    "mistral-7b": "qwen3-4b",
    "zephyr-7b": "qwen3-8b",
    "mistral-nemo": "qwen3-8b",
    "gpt-4o-mini": "qwen3-4b",
    "deepseek-chat": "qwen3-8b",
    "qwen": "qwen3-8b",
}

DEFAULT_MODELS = ["qwen3-4b", "qwen3-8b"]

# ---------------------------------------------------------------------------
# Judge models
# ---------------------------------------------------------------------------

_LOCAL_JUDGE_MODELS: dict[str, dict[str, Any]] = {
    "qwen3-4b": {
        "provider": "ollama",
        "model_id": QWEN3_4B_MODEL_ID,
        "display_name": "Qwen3 4B · локальный Judge",
    },
    "qwen3-8b": {
        "provider": "ollama",
        "model_id": QWEN3_8B_MODEL_ID,
        "display_name": "Qwen3 8B · локальный Judge",
    },
}

# Some existing Pydantic request schemas still have the old values as
# defaults (for example "mistral-7b" and "gpt-oss-120b"). Keep accepting
# them for backward compatibility, but resolve them to a real local model.
JUDGE_MODEL_ALIASES: dict[str, str] = {
    "gpt-oss-120b": "qwen3-8b",
    "llama-3.3-70b": "qwen3-8b",
    "qwen3-32b": "qwen3-8b",
    "mistral-7b": "qwen3-8b",
    "zephyr-7b": "qwen3-8b",
    "mistral-nemo": "qwen3-8b",
    "deepseek-chat": "qwen3-8b",
    "gpt-4o-mini": "qwen3-8b",
}

JUDGE_MODEL: dict[str, dict[str, Any]] = dict(_LOCAL_JUDGE_MODELS)
for _legacy_key, _canonical_key in JUDGE_MODEL_ALIASES.items():
    JUDGE_MODEL[_legacy_key] = {
        **_LOCAL_JUDGE_MODELS[_canonical_key],
        "alias_of": _canonical_key,
        "display_name": (
            f"{_LOCAL_JUDGE_MODELS[_canonical_key]['display_name']} "
            f"(совместимость: {_legacy_key})"
        ),
    }

# Only canonical models are shown in the UI. Legacy aliases remain accepted
# by the request schemas for backward compatibility.
PUBLIC_JUDGE_MODELS = dict(_LOCAL_JUDGE_MODELS)

DEFAULT_JUDGE = "qwen3-8b"


def resolve_judge_model_key(model_key: str | None) -> str:
    """Return a canonical Judge key, falling back to the configured default."""
    if not model_key or model_key not in JUDGE_MODEL:
        return DEFAULT_JUDGE
    return JUDGE_MODEL[model_key].get("alias_of", model_key)


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


def load_prompt(filename: str) -> str:
    prompt_dir = Path(__file__).resolve().parent.parent / "prompts"
    file_path = prompt_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    return file_path.read_text(encoding="utf-8").strip()


SYSTEM_PROMPT = load_prompt("system_prompt.md")
JUDGE_PROMPT_TEMPLATE = load_prompt("judge_prompt.md")


# ---------------------------------------------------------------------------
# Ollama inference
# ---------------------------------------------------------------------------


def _error_result(model_id: str, message: str) -> dict[str, str]:
    return {
        "model": model_id,
        "content": message,
        "status": "error",
    }


async def fetch_from_ollama(
        session: aiohttp.ClientSession,
        model_id: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        *,
        json_mode: bool = False,
) -> dict[str, str]:
    """Call the local Ollama /api/chat endpoint.

    The function never raises for ordinary provider/network/model errors. It
    returns a status=error result so the attempted Arena battle can still be
    persisted and shown to the user.
    """
    if not prompt or not prompt.strip():
        return _error_result(model_id, "Пустой промпт: запрос в Ollama не отправлен.")

    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": OLLAMA_THINK,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": float(temperature),
            "num_predict": int(max_tokens),
            "num_ctx": OLLAMA_NUM_CTX,
        },
    }
    if json_mode:
        # Ollama's constrained JSON output is useful for the Judge response.
        payload["format"] = "json"

    logger.info("🦙 Ollama запрос к %s (num_ctx=%s)", model_id, OLLAMA_NUM_CTX)
    started_at = time.perf_counter()

    try:
        async with OLLAMA_REQUEST_SEMAPHORE:
            async with session.post(
                    OLLAMA_CHAT_URL,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=OLLAMA_REQUEST_TIMEOUT),
            ) as response:
                response_text = await response.text()

                if response.status != 200:
                    detail = response_text[:700]
                    try:
                        error_data = json.loads(response_text)
                        detail = str(error_data.get("error", detail))
                    except (json.JSONDecodeError, AttributeError):
                        pass

                    lowered_detail = detail.lower()
                    if response.status == 404 or "model not found" in lowered_detail:
                        error_message = (
                            f"Модель Ollama '{model_id}' не найдена локально. "
                            f"Загрузите её командой: ollama pull {model_id}. "
                            f"Ответ Ollama: {detail}"
                        )
                    else:
                        error_message = (
                            f"Ollama вернул HTTP {response.status}: {detail}"
                        )

                    logger.error("❌ Ollama ошибка %s: %s", response.status, detail)
                    return _error_result(model_id, error_message)

                try:
                    data = json.loads(response_text)
                except json.JSONDecodeError as exc:
                    logger.error("Ollama вернул невалидный JSON: %s", response_text[:300])
                    return _error_result(
                        model_id,
                        f"Ollama вернул некорректный JSON: {exc}",
                    )

                message = data.get("message") or {}
                content = message.get("content")

                # Qwen3 may return a separate `thinking` field. We deliberately
                # use only `content` as the model's answer/evidence and never
                # persist the internal thinking field in Arena history.
                if not isinstance(content, str) or not content.strip():
                    logger.error(
                        "Ollama не вернул message.content. Ответ: %s",
                        response_text[:500],
                    )
                    return _error_result(
                        model_id,
                        "Ollama завершил запрос, но не вернул непустой message.content.",
                    )

                elapsed = time.perf_counter() - started_at
                logger.info(
                    "✅ Ollama успех: %s, %d символов, %.2f сек., prompt=%s tokens, output=%s tokens",
                    model_id,
                    len(content),
                    elapsed,
                    data.get("prompt_eval_count", "?"),
                    data.get("eval_count", "?"),
                )
                return {
                    "model": model_id,
                    "content": content.strip(),
                    "status": "success",
                }

    except asyncio.TimeoutError:
        logger.error("⏱️ Ollama TIMEOUT для модели %s", model_id)
        return _error_result(
            model_id,
            f"Тайм-аут Ollama: модель '{model_id}' не ответила за {OLLAMA_REQUEST_TIMEOUT:g} сек.",
        )
    except aiohttp.ClientConnectorError as exc:
        logger.exception("Не удалось подключиться к Ollama")
        return _error_result(
            model_id,
            f"Не удалось подключиться к Ollama по адресу {OLLAMA_BASE_URL}. "
            f"Проверьте, что приложение Ollama запущено. Детали: {exc}",
        )
    except aiohttp.ClientError as exc:
        logger.exception("Ошибка HTTP-клиента при обращении к Ollama")
        return _error_result(model_id, f"Сетевая ошибка Ollama: {exc}")
    except Exception as exc:  # Keep an external/model error as Arena evidence.
        logger.exception("Непредвиденная ошибка при обращении к Ollama")
        return _error_result(model_id, f"Ошибка Ollama: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Model routing and pairwise generation
# ---------------------------------------------------------------------------


async def fetch_from_model(
        session: aiohttp.ClientSession,
        model_key: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2000,
) -> dict[str, str]:
    """Resolve a public/legacy model key and call the corresponding Ollama tag."""
    canonical_key = LEGACY_MODEL_ALIASES.get(model_key, model_key)
    config = AVAILABLE_MODELS.get(canonical_key)

    if config is None:
        logger.error("Модель %s не найдена в AVAILABLE_MODELS", model_key)
        return _error_result(
            model_key,
            f"Модель '{model_key}' не зарегистрирована в Arena. Обновите список моделей.",
        )

    model_id = config["model_id"]
    return await fetch_from_ollama(
        session=session,
        model_id=model_id,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )


async def compare_models(
        models: list[str],
        session: aiohttp.ClientSession,
        custom_prompt: str | None = None,
) -> dict[str, Any]:
    """Run participant models sequentially and preserve every attempt as evidence.

    Sequential calls are intentional: two different GGUF models should not
    compete for the laptop's 8 GB VRAM at the same time.
    """
    prompt = custom_prompt or SYSTEM_PROMPT

    if not models:
        return {"error": "Не выбрано ни одной модели", "results": []}

    logger.info("📊 Последовательный запуск сравнения %d модели(й)", len(models))
    results: list[dict[str, str]] = []

    for model_key in models:
        result = await fetch_from_model(
            session=session,
            model_key=model_key,
            prompt=prompt,
            temperature=0.0,
            max_tokens=2000,
        )
        # Do not drop failures. They are part of the battle history and are
        # consumed by judge_winner() to produce an availability verdict.
        results.append(result)

    return {"results": results}


# ---------------------------------------------------------------------------
# Judge response parsing
# ---------------------------------------------------------------------------


def extract_json(content: str) -> dict[str, Any]:
    """Extract the first JSON object containing `winner` and `reason`.

    Handles plain JSON, Markdown code fences, and short text preceding JSON.
    """
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Пустой ответ Judge")

    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            parsed, _ = decoder.raw_decode(cleaned[match.start():])
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict) and "winner" in parsed and "reason" in parsed:
            return parsed

    raise ValueError("JSON-объект с полями winner и reason не найден")


REQUIRED_PLACEHOLDERS = (
    "{model_a_name}",
    "{response_a}",
    "{model_b_name}",
    "{response_b}",
)


async def ask_judge(
        session: aiohttp.ClientSession,
        model1: str,
        response1: str,
        model2: str,
        response2: str,
        judge_model_key: str | None = None,
        judge_prompt: str | None = None,
) -> dict[str, Any]:
    """Send two model responses to the configured local Ollama Judge."""
    canonical_judge_key = resolve_judge_model_key(judge_model_key or DEFAULT_JUDGE)

    if judge_prompt is not None:
        if not judge_prompt.strip():
            logger.warning("Пустой judge_prompt; используется шаблон по умолчанию")
            judge_prompt = None
        else:
            missing = [placeholder for placeholder in REQUIRED_PLACEHOLDERS if placeholder not in judge_prompt]
            if missing:
                message = (
                        "Кастомный judge_prompt не содержит обязательные плейсхолдеры: "
                        + ", ".join(missing)
                )
                logger.error(message)
                return {"error": message}

    # Escape braces in model responses so that JSON/code examples in them are
    # not interpreted as Python str.format placeholders.
    safe_response1 = str(response1).replace("{", "{{").replace("}", "}}")
    safe_response2 = str(response2).replace("{", "{{").replace("}", "}}")

    template = judge_prompt or JUDGE_PROMPT_TEMPLATE
    try:
        prompt = template.format(
            model_a_name=model1,
            response_a=safe_response1,
            model_b_name=model2,
            response_b=safe_response2,
        )
    except (KeyError, ValueError, IndexError) as exc:
        logger.exception("Ошибка форматирования judge_prompt")
        return {"error": f"Ошибка форматирования judge_prompt: {exc}"}

    judge_config = JUDGE_MODEL[canonical_judge_key]
    judge_model_id = judge_config["model_id"]
    logger.info(
        "⚖️ Отправка запроса локальному Judge %s (%s)",
        canonical_judge_key,
        judge_model_id,
    )

    response = await fetch_from_ollama(
        session=session,
        model_id=judge_model_id,
        prompt=prompt,
        temperature=0.0,
        max_tokens=1500,
        json_mode=True,
    )

    if response["status"] != "success":
        error_message = f"Локальный Judge не ответил: {response['content']}"
        logger.error(error_message)
        return {"error": error_message}

    try:
        verdict = extract_json(response["content"])
    except (ValueError, TypeError) as exc:
        logger.warning("Ошибка парсинга JSON ответа Judge: %s", exc)
        return {
            "error": f"Judge вернул некорректный JSON: {exc}",
            "raw_response": response["content"],
        }

    winner = str(verdict.get("winner", "")).strip().upper()
    reason = verdict.get("reason")

    if winner not in {"MODEL_A", "MODEL_B", "DRAW"}:
        return {
            "error": f"Judge вернул недопустимое значение winner: {winner!r}",
            "raw_response": response["content"],
        }

    if not isinstance(reason, str) or not reason.strip():
        return {
            "error": "Judge не вернул непустое поле reason",
            "raw_response": response["content"],
        }

    logger.info("✅ Локальный Judge вернул: %s", winner)
    return {"winner": winner, "reason": reason.strip()}


# ---------------------------------------------------------------------------
# Arena verdict logic
# ---------------------------------------------------------------------------


async def judge_winner(
        results: list[dict[str, Any]],
        session: aiohttp.ClientSession,
        judge_model: str | None = None,
        judge_prompt: str | None = None,
) -> dict[str, Any]:
    """Produce an Arena verdict, including technical unavailability verdicts.

    Business rule preserved: an attempted battle is not discarded when a
    participant model is unavailable. With one successful participant, the
    available model is returned as the technical fallback winner and the user
    is told to replace the unavailable model. If both fail, there is no winner.
    """
    canonical_judge_key = resolve_judge_model_key(judge_model or DEFAULT_JUDGE)
    logger.info("⚖️ Начало судейства, модель: %s", canonical_judge_key)

    successful_results = [
        item for item in results if item.get("status") == "success"
    ]
    failed_results = [
        item for item in results if item.get("status") != "success"
    ]

    if not successful_results:
        logger.error("❌ Ни одна из моделей не вернула ответ")
        failure_details = "; ".join(
            f"{item.get('model', 'unknown')}: {item.get('content', 'нет подробностей')}"
            for item in failed_results
        )
        reason = (
            "Не удалось получить ответ ни от одной выбранной модели. "
            "Проверьте, что Ollama запущен, а модели загружены локально; "
            "затем выберите другие модели."
        )
        if failure_details:
            reason = f"{reason} Подробности: {failure_details}"

        return {
            "winners": [],
            "losers": [item.get("model", "unknown") for item in failed_results],
            "message": (
                "❌ Выбранные модели недоступны или не смогли ответить. "
                "Проверьте Ollama либо выберите другие модели."
            ),
            "judge_result": {"winner": None, "reason": reason},
            "reason": reason,
            "evidence": results,
            "winner_position": None,
            "judge_model": canonical_judge_key,
        }

    if len(successful_results) == 1:
        winner = successful_results[0]
        failed_names = [item.get("model", "unknown") for item in failed_results]
        failed_details = "; ".join(
            f"{item.get('model', 'unknown')}: {item.get('content', 'нет подробностей')}"
            for item in failed_results
        )
        if failed_names:
            unavailable_text = ", ".join(failed_names)
            reason = (
                f"Модель(и) {unavailable_text} не ответила(и) из-за технической ошибки. "
                "Выберите другую модель вместо недоступной. "
                f"Единственный полученный ответ: {winner.get('model', 'unknown')}."
            )
            if failed_details:
                reason = f"{reason} Подробности: {failed_details}"
            message = (
                f"⚠️ Модель(и) {unavailable_text} недоступна(ы) — выберите другую. "
                f"Ответ получен только от {prettify_model_name(winner.get('model', 'unknown'))}."
            )
        else:
            reason = "Получен только один успешный ответ; полноценное сравнение невозможно."
            message = "⚠️ Получен только один ответ; полноценное сравнение невозможно."

        winner_index = next(
            (index for index, item in enumerate(results) if item is winner),
            0,
        )
        winner_position = "MODEL_A" if winner_index == 0 else "MODEL_B" if winner_index == 1 else None
        winner_model = winner.get("model", "unknown")

        logger.info("⚠️ Технический вердикт: доступна только модель %s", winner_model)
        return {
            "winners": [winner_model],
            "losers": failed_names,
            "message": message,
            "judge_result": {
                "winner": winner_model,
                "reason": reason,
                "automatic": True,
            },
            "reason": reason,
            "evidence": results,
            "winner_position": winner_position,
            "judge_model": canonical_judge_key,
        }

    # The Arena is a pairwise comparison. If a client submits more than two
    # models, compare the first two successful responses as before.
    res1, res2 = successful_results[0], successful_results[1]
    model1 = res1.get("model", "MODEL_A")
    model2 = res2.get("model", "MODEL_B")
    response1 = res1.get("content", "")
    response2 = res2.get("content", "")

    if model1 == model2:
        judge_model1 = f"{model1} (MODEL_A)"
        judge_model2 = f"{model2} (MODEL_B)"
    else:
        judge_model1, judge_model2 = model1, model2

    judge_result = await ask_judge(
        session=session,
        model1=judge_model1,
        response1=response1,
        model2=judge_model2,
        response2=response2,
        judge_model_key=canonical_judge_key,
        judge_prompt=judge_prompt,
    )

    if "error" in judge_result:
        error_detail = judge_result.get("error", "Неизвестная ошибка")
        logger.error("❌ Ошибка Judge: %s", error_detail)
        return {
            "winners": [],
            "losers": [],
            "message": "❌ Локальный Judge не смог определить победителя.",
            "judge_result": {"winner": None, "reason": error_detail},
            "judge_error": judge_result,
            "evidence": results,
            "winner_position": None,
            "judge_model": canonical_judge_key,
            "reason": error_detail,
        }

    winner_alias = judge_result["winner"]
    if winner_alias == "DRAW":
        reason = judge_result.get("reason", "")
        logger.info("🤝 Judge зафиксировал ничью")
        return {
            "winners": [],
            "losers": [],
            "message": "🤝 Ничья!",
            "judge_result": {"winner": "DRAW", "reason": reason},
            "reason": reason,
            "evidence": results,
            "winner_position": None,
            "judge_model": canonical_judge_key,
        }

    winner_position = winner_alias
    winner_map = {"MODEL_A": model1, "MODEL_B": model2}
    winner = winner_map[winner_alias]
    losers = [model for model in (model1, model2) if model != winner]

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
        "judge_model": canonical_judge_key,
    }


# ---------------------------------------------------------------------------
# Public Arena orchestration function
# ---------------------------------------------------------------------------


async def run_arena_comparison(
        models: list[str],
        session: aiohttp.ClientSession,
        prompt: str | None = None,
) -> dict[str, Any]:
    """Run participant models and return evidence plus elapsed time."""
    started_at = time.perf_counter()
    compare_result = await compare_models(
        models=models,
        session=session,
        custom_prompt=prompt,
    )
    elapsed = round(time.perf_counter() - started_at, 2)

    if "error" in compare_result and not compare_result.get("results"):
        return {
            "error": compare_result["error"],
            "results": [],
            "elapsed": elapsed,
        }

    return {
        "results": compare_result.get("results", []),
        "elapsed": elapsed,
    }
