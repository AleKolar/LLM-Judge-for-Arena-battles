# src/services/download_service.py
from src.models.db_models import ArenaResult
from src.utils.normalize import normalize_evidence, to_md
from src.utils.prettify_model_name import prettify_model_name


def generate_battle_markdown(battle: ArenaResult) -> str:
    evidence = normalize_evidence(battle.evidence)
    evidence_md = to_md(evidence)

    # Победитель
    winner_text = "Ничья"
    if battle.winner:
        winner_text = prettify_model_name(battle.winner)
        if battle.model1 == battle.model2:
            label = "Модель A" if battle.winner_position == "MODEL_A" else "Модель B"
            winner_text += f" ({label})"

    verdict = battle.message or "Результат не определён"
    reason = battle.judge_reason or "Комментарий отсутствует"

    # Имена для таблицы статусов
    model1_display = battle.model1
    model2_display = battle.model2
    if battle.model1 == battle.model2:
        model1_display += " (Модель A)"
        model2_display += " (Модель B)"

    # Статусы по winner_position
    if battle.winner and battle.winner_position:
        if battle.winner_position == "MODEL_A":
            status1, status2 = "✅", "❌"
        else:
            status1, status2 = "❌", "✅"
    else:
        status1 = status2 = "🤝"

    return (
        f"# 🧠 LLM Arena — Результат битвы\n\n"
        f"## 🏆 Победитель\n{winner_text}\n\n"
        f"## ⚖️ Вердикт Судьи\n{verdict}\n\n"
        f"## 📋 Комментарии к результату\n{reason}\n\n"
        f"---\n\n"
        f"## 📊 Статус моделей\n\n"
        f"| Модель | Статус |\n"
        f"|--------|--------|\n"
        f"| {model1_display} | {status1} |\n"
        f"| {model2_display} | {status2} |\n\n"
        f"---\n\n"
        f"## 📝 Код моделей\n\n"
        f"{evidence_md}"
    )