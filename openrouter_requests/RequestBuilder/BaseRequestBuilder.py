from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from openrouter_requests.schemas import OpenrouterRequest

class BaseRequestBuilder(ABC):

    @abstractmethod
    async def build_request(
            self,
            data: OpenrouterRequest) -> Dict[str, Any]:
        pass