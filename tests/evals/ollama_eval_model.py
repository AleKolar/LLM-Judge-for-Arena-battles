"""DeepEval Ollama evaluator with Qwen3 thinking disabled.

The stock DeepEval OllamaModel forwards generation_kwargs to Ollama's
`options` object. Ollama's `think` flag is a top-level chat argument, so it
must be passed separately to `chat()` for Qwen3 models.
"""

from deepeval.models import OllamaModel


class ThinkingDisabledOllamaModel(OllamaModel):
    """OllamaModel that explicitly sends ``think=False`` on async requests."""

    async def a_generate(self, prompt: str, schema=None, audio=None):
        """Generate a response asynchronously in DeepEval's expected format."""
        chat_model = self.load_model(async_mode=True)
        messages = self._prompt_messages(prompt, audio)

        response = await chat_model.chat(
            model=self.name,
            messages=messages,
            think=False,
            format=schema.model_json_schema() if schema else None,
            options={
                "temperature": self.temperature,
                **self.generation_kwargs,
            },
        )

        content = (response.message.content or "").strip()
        if not content:
            raise ValueError(
                f"Ollama returned empty message.content for evaluator model {self.name!r}."
            )

        output = schema.model_validate_json(content) if schema else content
        return output, 0
