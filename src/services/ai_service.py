# src/services/ai_service.py
import asyncio
import os
import re
import time
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

AVAILABLE_MODELS = {
    # Старые (проверенные)
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "deepseek-chat": "deepseek/deepseek-chat",
    "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct",

    # Новые (Оч.сильные)
    "qwen3-coder-480b": "qwen/qwen3-coder-480b-a35b-instruct:free",
    "llama-3.3-70b": "meta-llama/llama-3.3-70b-instruct:free",

    # Средние (не проверенные)
    "llama-3.2-3b": "meta-llama/llama-3.2-3b-instruct:free",
}

DEFAULT_MODELS = ["gpt-4o-mini", "deepseek-chat"]

JUDGE_MODEL = "deepseek-chat"

def load_prompt(filename: str) -> str:
    """Читает промпт из файла в папке prompts."""
    prompt_dir = Path(__file__).resolve().parent.parent / "prompts"
    file_path = prompt_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    return file_path.read_text(encoding="utf-8").strip()

SYSTEM_PROMPT = load_prompt("system_prompt.md")

JUDGE_PROMPT_TEMPLATE = load_prompt("judge_prompt.md")

async def fetch_from_model(session: aiohttp.ClientSession, model_id: str, prompt: str) -> dict:
    if not API_KEY:
        return {"model": model_id, "content": "Ошибка: API-ключ не задан", "status": "error"}

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 500,
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
            else:
                error_text = await resp.text()
                return {
                    "model": model_id,
                    "content": f"Ошибка {resp.status}: {error_text[:200]}",
                    "status": "error",
                }
    except Exception as e:
        return {"model": model_id, "content": f"Исключение: {str(e)}", "status": "error"}

async def compare_models(
    models: list[str],
    session: aiohttp.ClientSession,
    custom_prompt: str = None
) -> dict:
    prompt = custom_prompt or SYSTEM_PROMPT
    selected_ids = [AVAILABLE_MODELS[m] for m in models if m in AVAILABLE_MODELS]
    if not selected_ids:
        return {"error": "Не выбрано ни одной модели"}

    tasks = [fetch_from_model(session, mid, prompt) for mid in selected_ids]
    results = await asyncio.gather(*tasks)
    return {"results": results}


def split_code_and_tests(content: str) -> tuple[str, str]:
    """Разделяет код на функцию и тесты."""
    lines = content.split('\n')
    func_lines = []
    test_lines = []
    in_func = False
    in_tests = False
    
    for line in lines:
        if line.startswith('def is_leap_year'):
            in_func = True
        if in_func and line.strip().startswith('def test_'):
            in_tests = True
            in_func = False
        if in_func:
            func_lines.append(line)
        elif in_tests:
            test_lines.append(line)
    
    return '\n'.join(func_lines), '\n'.join(test_lines)


async def ask_judge(
    session: aiohttp.ClientSession,
    model1: str,
    code1: str,
    tests1: str,
    model2: str,
    code2: str,
    tests2: str
) -> dict:
    """Отправляет запрос Судье (deepseek-chat) для определения победителя."""
    import json

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        model1=model1,
        code1=code1,
        tests1=tests1,
        model2=model2,
        code2=code2,
        tests2=tests2
    )

    judge_id = AVAILABLE_MODELS[JUDGE_MODEL]
    response = await fetch_from_model(session, judge_id, prompt)

    if response["status"] != "success":
        return {"error": f"Судья не ответил: {response['content']}"}

    try:
        verdict = json.loads(response["content"])
        winner = verdict.get("winner")
        reason = verdict.get("reason", "Без обоснования")
        if winner not in [model1, model2]:
            return {"error": f"Неверный победитель: {winner}"}
        return {"winner": winner, "reason": reason}
    except json.JSONDecodeError:
        return {"error": f"Неверный формат ответа Судьи: {response['content']}"}


def analyze_tests(code: str) -> dict:
    """Оценивает качество тестов на основе покрытия ключевых случаев и стиля."""

    # Количество assert
    num_asserts = len(re.findall(r'\bassert\b', code))

    # Проверка наличия тестов для ключевых годов
    has_1900 = bool(re.search(r'\b1900\b', code))
    has_2000 = bool(re.search(r'\b2000\b', code))
    has_2100 = bool(re.search(r'\b2100\b', code))
    has_negative = bool(re.search(r'-\d{1,4}(?!\d)', code))  # отрицательный год
    has_typical_leap = bool(re.search(r'\b(2024|2020|2004)\b', code))  # любой типичный високосный
    has_typical_common = bool(re.search(r'\b(2023|2021|2003)\b', code))  # любой типичный невисокосный

    # Бонус за стиль: сообщения в ассертах
    has_assert_messages = bool(re.search(r'assert .+,\s*["\']', code))

    # Суммарный охват ключевых случаев
    coverage_points = sum([
        has_1900, has_2000, has_2100, has_negative,
        has_typical_leap, has_typical_common
    ])

    return {
        "num_asserts": num_asserts,
        "has_1900": has_1900,
        "has_2000": has_2000,
        "has_2100": has_2100,
        "has_negative": has_negative,
        "has_typical_leap": has_typical_leap,
        "has_typical_common": has_typical_common,
        "has_assert_messages": has_assert_messages,
        "coverage_points": coverage_points,
    }


async def judge_winner(results: list[dict], session: aiohttp.ClientSession) -> dict:
    winners, losers = [], []
    evidence = []
    pattern = re.compile(r'(year\s*%\s*400\s*==\s*0|not\s+year\s*%\s*400|year\s*%\s*400\b)')

    for res in results:
        code = res.get("content", "")
        model = res.get("model", "unknown")
        status = res.get("status", "success")

        # Если модель вернула ошибку – сразу в проигравшие, без анализа правила
        if status == "error":
            evidence.append({
                "model": model,
                "found_rule": False,
                "snippet": "",
                "status": "error",
                "num_asserts": 0,
                "coverage_points": 0,
                "has_assert_messages": False,
                "has_1900": False,
                "has_2000": False,
                "has_2100": False,
                "has_negative": False,
                "has_typical_leap": False,
                "has_typical_common": False,
            })
            losers.append(model)
            continue

        # Нормальный ответ – анализируем правило и тесты
        match = pattern.search(code)
        found_rule = bool(match)
        snippet = match.group(0) if match else ""
        test_stats = analyze_tests(code)  # функция возвращает dict с ключами: num_asserts, coverage_points, has_*, ...

        evidence.append({
            "model": model,
            "found_rule": found_rule,
            "snippet": snippet,
            "status": "success",
            **test_stats
        })

        if found_rule:
            winners.append(model)
        else:
            losers.append(model)

    # Определение победителя
    successful_results = [r for r in results if r.get("status") == "success"]
    if len(successful_results) == 2:
        # Две успешные модели — Судья решает
        res1, res2 = successful_results
        code1 = res1["content"]
        code2 = res2["content"]
        model1 = res1["model"]
        model2 = res2["model"]
        
        # Разделим код на функцию и тесты
        func1, tests1 = split_code_and_tests(code1)
        func2, tests2 = split_code_and_tests(code2)
        
        judge_result = await ask_judge(session, model1, func1, tests1, model2, func2, tests2)
        
        if "error" in judge_result:
            # Fallback к старой логике
            return await judge_winner_fallback(results)
        
        winner = judge_result["winner"]
        reason = judge_result["reason"]
        msg = f"🏆 Судья решил: {winner} побеждает! Причина: {reason}"
        final_winners = [winner]
        losers = [m for m in [model1, model2] if m != winner]
    else:
        # Старая логика для других случаев
        return await judge_winner_fallback(results)

    return {
        "winners": final_winners,
        "losers": losers,
        "message": msg,
        "evidence": evidence
    }


async def judge_winner_fallback(results: list[dict]) -> dict:
    winners, losers = [], []
    evidence = []
    pattern = re.compile(r'(year\s*%\s*400\s*==\s*0|not\s+year\s*%\s*400|year\s*%\s*400\b)')

    for res in results:
        code = res.get("content", "")
        model = res.get("model", "unknown")
        status = res.get("status", "success")

        # Если модель вернула ошибку – сразу в проигравшие, без анализа правила
        if status == "error":
            evidence.append({
                "model": model,
                "found_rule": False,
                "snippet": "",
                "status": "error",
                "num_asserts": 0,
                "coverage_points": 0,
                "has_assert_messages": False,
                "has_1900": False,
                "has_2000": False,
                "has_2100": False,
                "has_negative": False,
                "has_typical_leap": False,
                "has_typical_common": False,
            })
            losers.append(model)
            continue

        # Нормальный ответ – анализируем правило и тесты
        match = pattern.search(code)
        found_rule = bool(match)
        snippet = match.group(0) if match else ""
        test_stats = analyze_tests(code)

        evidence.append({
            "model": model,
            "found_rule": found_rule,
            "snippet": snippet,
            "status": "success",
            **test_stats
        })

        if found_rule:
            winners.append(model)
        else:
            losers.append(model)

    # Определение победителя
    if not winners:
        msg = f"😞 Все проиграли: {', '.join(losers)}. Никто не учёл правило 400."
        final_winners = []
    elif len(winners) == 1:
        msg = f"🏆 Победитель: {winners[0]}! Учтено правило 400."
        final_winners = [winners[0]]
    else:
        # Ничья по функции – сравниваем качество тестов
        winner_ev = [e for e in evidence if e["model"] in winners]
        best = sorted(winner_ev, key=lambda e: (
            e["coverage_points"],
            e["has_assert_messages"],
            e["num_asserts"]
        ), reverse=True)

        if (best[0]["coverage_points"] > best[1]["coverage_points"]) or \
           (best[0]["coverage_points"] == best[1]["coverage_points"] and
            best[0]["has_assert_messages"] and not best[1]["has_assert_messages"]):
            champ = best[0]["model"]
            msg = f"🏆 По функции ничья, но {champ} побеждает за счёт лучших тестов! (покрытие: {best[0]['coverage_points']}/6)"
            final_winners = [champ]
        else:
            msg = f"🤝 Абсолютная ничья! Модели {', '.join(winners)} справились с задачей и тестами."
            final_winners = winners

    return {
        "winners": final_winners,
        "losers": losers,
        "message": msg,
        "evidence": evidence
    }

async def run_arena_comparison(
    models: list[str],
    session: aiohttp.ClientSession,
    prompt: str = None
) -> dict:
    """Оборачивает сравнение моделей и добавляет время выполнения."""
    start = time.time()
    result = await compare_models(models, session, prompt)
    elapsed = time.time() - start
    result["elapsed"] = round(elapsed, 2)
    return result

