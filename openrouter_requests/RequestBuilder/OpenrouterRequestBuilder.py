from typing import Any, Dict, List, Optional
from openrouter_requests.RequestBuilder.BaseRequestBuilder import BaseRequestBuilder


class OpenrouterRequestBuilder(BaseRequestBuilder):

    def __init__(self) -> None:
        pass

    async def build_request(
            self,
            model: str,
            context: List[Dict[str, str]],
            tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return {
            "model": model,
            "messages": context,
            "tools": tools or [],
        }