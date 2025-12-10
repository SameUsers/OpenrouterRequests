from typing import Any, Dict, Optional

import httpx

from openrouter_requests.TransportModule.BaseTransport import Transport


class HttpxProcessor(Transport):

    def __init__(self, request_timeout: int = 30) -> None:
        self._client = httpx.AsyncClient(timeout=request_timeout)

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