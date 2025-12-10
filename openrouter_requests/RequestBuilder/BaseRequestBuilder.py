from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseRequestBuilder(ABC):

    @abstractmethod
    async def build_request(
            self,
            model: str,
            context: List[Dict[str, str]],
            tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        pass