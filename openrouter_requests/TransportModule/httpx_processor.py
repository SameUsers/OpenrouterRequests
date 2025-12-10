from typing import Any, Dict, Optional
import httpx
import threading
from loguru import logger
from openrouter_requests.TransportModule.BaseTransport import Transport

transport = httpx.AsyncHTTPTransport(
    retries=2,
)

limits = httpx.Limits(
    max_connections=100,
    max_keepalive_connections=20,
    keepalive_expiry=30,
)

timeout = httpx.Timeout(
    connect=2.0,
    read=10.0,
    write=5.0,
    pool=1.0,
)


class HttpxProcessor(Transport):
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, '_initialized') or not self._initialized:
            self._client = httpx.AsyncClient(timeout=timeout,
                                             limits=limits,
                                             transport=transport,
                                             http2=True)
            self._initialized = True
            logger.success(
                "Инициализирован синглтон класса {} с параметрками {}",
                self.__class__.__name__,
                self.__dict__
            )

    async def get(
            self,
            url: str,
            headers: Dict[str, str],
            payload: Optional[Any],
    ) -> Any:
        response = await self._client.get(
            url=url,
            headers=headers,
            params=payload,
        )
        response.raise_for_status()
        return response.text

    async def post(
            self,
            url: str,
            headers: Dict[str, str],
            payload: Optional[Any],
    ) -> Any:
        response = await self._client.post(
            url=url,
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        return response.json()