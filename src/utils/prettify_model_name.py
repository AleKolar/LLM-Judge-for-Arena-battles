# src/utils/prettify_model_name.py
def prettify_model_name(model_id: str) -> str:
    return model_id.split("/")[-1]