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