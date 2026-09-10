from typing import Protocol

from openai import OpenAI


class EmbeddingProvider(Protocol):
    @property
    def identifier(self) -> str: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class OpenAIEmbeddingProvider:
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        client: OpenAI | None = None,
    ) -> None:
        self._model = model
        self._client = client or OpenAI()

    @property
    def identifier(self) -> str:
        return f"openai:{self._model}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [
            item.embedding
            for item in sorted(response.data, key=lambda item: item.index)
        ]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
