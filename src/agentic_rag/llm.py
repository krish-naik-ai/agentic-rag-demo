from typing import Protocol

from openai import OpenAI


class LanguageModel(Protocol):
    def complete_json(self, *, system_prompt: str, user_prompt: str) -> str: ...


class OpenAIChatModel:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        client: OpenAI | None = None,
    ) -> None:
        self._model = model
        self._client = client or OpenAI()

    def complete_json(self, *, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("OpenAI returned an empty response.")
        return content
