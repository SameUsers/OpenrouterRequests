from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional, Sequence
import threading
import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from loguru import logger


class ChromaVectorStore:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(
        self,
        collection_name: str = "default_docs",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        client: Optional["ClientAPI"] = None,
    ) -> None:
        if self._initialized:
            return

        if client is None:
            client = chromadb.PersistentClient(
                path="./chroma_db",
                settings=Settings(allow_reset=True),
            )

        self._client: "ClientAPI" = client
        self._collection: "Collection" = self._client.get_or_create_collection(
            name=collection_name,
        )
        self._model = SentenceTransformer(model_name, device="cpu")
        self._initialized = True
        logger.success(
            "Инициализирован синглтон класса {} с параметрками {}",
            self.__class__.__name__,
            self.__dict__
        )


    async def add_document(
            self,
            doc_id: str,
            text: str,
            metadata: Optional[Dict[str, Any]] = None,
    ) -> None:

        await self.add_documents(
            ids=[doc_id],
            texts=[text],
            metadatas=[metadata] if metadata is not None else None,
        )

    async def add_documents(
            self,
            ids: Sequence[str],
            texts: Sequence[str],
            metadatas: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    ) -> None:

        if len(ids) != len(texts):
            raise ValueError("Длины ids и texts должны совпадать")

        if metadatas is not None and len(metadatas) != len(ids):
            raise ValueError("Длина metadatas должна соответствовать длине ids")

        if metadatas is not None:
            metadatas_list: Optional[List[Dict[str, Any]]] = [
                self._normalize_metadata(m) for m in metadatas
            ]
        else:
            metadatas_list = None

        embeddings = await self._embed_batch(texts)

        self._collection.add(
            ids=list(ids),
            documents=list(texts),
            embeddings=embeddings,
            metadatas=metadatas_list,
        )

    async def search(
            self,
            query: str,
            k: int = 15,
    ) -> List[Dict[str, Any]]:

        query_embedding = await self._embed(query)

        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )

        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        items: List[Dict[str, Any]] = []
        for doc_id, text, metadata, distance in zip(
                ids,
                docs,
                metadatas,
                distances,
        ):
            items.append(
                {
                    "id": doc_id,
                    "text": text,
                    "metadata": metadata or {},
                    "score": float(distance),
                },
            )
        return items

    async def _embed(self, text: str) -> List[float]:

        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self._model.encode(
                text,
                show_progress_bar=False,
                normalize_embeddings=True,
            ).tolist(),
        )
        return embedding

    async def _embed_batch(self, texts: Sequence[str]) -> List[List[float]]:

        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._model.encode(
                list(texts),
                show_progress_bar=False,
                normalize_embeddings=True,
            ).tolist(),
        )
        return embeddings

    @staticmethod
    def _normalize_metadata(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:

        if not meta:
            return {}

        normalized: Dict[str, Any] = {}
        for key, value in meta.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                normalized[key] = value
            else:
                normalized[key] = json.dumps(value, ensure_ascii=False)
        return normalized