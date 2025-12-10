from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class Transport(ABC):

    @abstractmethod
    async def get(
            self,
            url: str,
            headers: Dict[str, str],
            payload: Optional[Any],
    ) -> Any:
        pass

    @abstractmethod
    async def post(
            self,
            url: str,
            headers: Dict[str, str],
            payload: Optional[Any],
    ) -> Any:
        pass