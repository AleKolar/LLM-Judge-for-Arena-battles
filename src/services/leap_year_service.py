from datetime import datetime

# Хранилище статистики (глобальное, можно заменить на БД)
stats: dict = {
    "total_checks": 0,
    "leap_count": 0,
    "common_count": 0,
    "recent_years": [],
    "celebrity_birthdays": {
        1964: "Киану Ривз",
        1975: "Анджелина Джоли",
        1984: "Скарлетт Йоханссон",
        2000: "Джаред Лето"
    }
}

LEAP_FACTS = {
    1900: "1900 — исключение: не был високосным! Парижская выставка, но без 29 февраля.",
    2000: "2000 — редкий високосный! Проблема Y2K оказалась раздута.",
    2020: "2020 — високосный, но мир запомнил его по другой причине...",
    2024: "2024 — у нас есть 29 февраля, наслаждаемся!"
}

def is_leap_year(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

def get_year_description(year: int) -> str:
    if year in LEAP_FACTS:
        return LEAP_FACTS[year]
    if is_leap_year(year):
        return f"{year} — високосный год! 366 дней, можно успеть больше!"
    if year == 1900:
        return "1900 — не високосный, хотя кратен 4! Исключение правил."
    return f"{year} — обычный год, 365 дней стабильности."

def record_check(year: int, leap: bool) -> None:
    stats["total_checks"] += 1
    if leap:
        stats["leap_count"] += 1
    else:
        stats["common_count"] += 1
    stats["recent_years"].append({
        "year": year, "is_leap": leap, "time": datetime.now().isoformat()
    })
    if len(stats["recent_years"]) > 10:
        stats["recent_years"].pop(0)

def get_stats() -> dict:
    return stats

def get_celebrity(year: int) -> str | None:
    return stats["celebrity_birthdays"].get(year)