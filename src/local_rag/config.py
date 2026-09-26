from __future__ import annotations

import os
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Settings:
    qdrant_url: str = "http://localhost:6333"
    collection: str = "documents"
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    api_key: str | None = None
    chat_model: str = "qwen2.5:7b"
    embedding_model: str = "nomic-embed-text"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            qdrant_url=os.getenv("LOCAL_RAG_QDRANT_URL", cls.qdrant_url),
            collection=os.getenv("LOCAL_RAG_COLLECTION", cls.collection),
            provider=os.getenv("LOCAL_RAG_PROVIDER", cls.provider),
            base_url=os.getenv("LOCAL_RAG_BASE_URL", cls.base_url),
            api_key=os.getenv("LOCAL_RAG_API_KEY"),
            chat_model=os.getenv("LOCAL_RAG_CHAT_MODEL", cls.chat_model),
            embedding_model=os.getenv(
                "LOCAL_RAG_EMBEDDING_MODEL", cls.embedding_model
            ),
        )

    def override(self, **values: object) -> "Settings":
        return replace(self, **{key: value for key, value in values.items() if value})

