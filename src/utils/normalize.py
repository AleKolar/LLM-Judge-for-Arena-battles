# src/utils/normalize.py
def to_md(evidence):
    blocks = []

    for item in evidence:
        blocks.append(
            f"## Model: {item.get('model', 'unknown')}\n\n"
            f"Status: {item.get('status', 'unknown')}\n\n"
            f"```python\n{item.get('content', '')}\n```\n"
        )

    return "\n".join(blocks)

def normalize_evidence(evidence):
    normalized = []

    for item in evidence:

        if isinstance(item, dict):
            normalized.append({
                "model": item.get("model", "unknown"),
                "status": item.get("status", "success"),
                "content": item.get("content", str(item)),
            })

        else:
            normalized.append({
                "model": "unknown",
                "status": "success",
                "content": str(item),
            })

    return normalized


def normalize_decision(decision: dict) -> dict:
    """
    Приводит ответ judge к единому формату WinnerResponse-safe.
    """

    if not isinstance(decision, dict):
        return {
            "winners": [],
            "losers": [],
            "message": "Invalid judge response",
            "reason": "Judge returned non-dict response",
            "judge_result": {"winner": None, "reason": "invalid"},
            "judge_error": {"error": "invalid format"},
            "winner_position": None,
        }

    return {
        "winners": decision.get("winners", []),
        "losers": decision.get("losers", []),
        "message": decision.get("message", ""),
        "summary": decision.get("summary"),
        "reason": decision.get("reason", ""),
        "judge_result": decision.get("judge_result", {}),
        "judge_error": decision.get("judge_error"),
        "winner_position": decision.get("winner_position"),
    }

