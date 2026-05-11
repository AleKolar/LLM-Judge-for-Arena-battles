# src/utils/normalize.py
def to_md(evidence):
    return "\n\n".join(
        f"### {e['model']}\n\n```python\n{e['content']}\n```"
        for e in evidence
    )

def normalize_evidence(evidence):
    if not evidence:
        return []

    normalized = []

    for item in evidence:
        if isinstance(item, dict):
            normalized.append({
                "model": item.get("model", "unknown"),
                "content": item.get("content", "")
            })
        else:
            normalized.append({
                "model": "unknown",
                "content": str(item)
            })

    return normalized