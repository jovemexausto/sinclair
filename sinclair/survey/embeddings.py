from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_openai import OpenAIEmbeddings


class EmbeddingBackend(ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbeddingBackend(EmbeddingBackend):
    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
    ) -> None:
        kwargs: dict[str, object] = {"model": model}
        if dimensions is not None:
            kwargs["dimensions"] = dimensions
        self._client = OpenAIEmbeddings(**kwargs)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [list(vector) for vector in self._client.embed_documents(texts)]
