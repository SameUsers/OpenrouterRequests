from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseContextManager(ABC):

    @abstractmethod
    async def add_message(self, message: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    async def get_context(self) -> List[Dict[str, Any]]:
        pass

    async def add_to_context(
            self,
            data: str,
            role: str,
            **extra: Any,
    ) -> None:
        message: Dict[str, Any] = {"role": role, "content": data}
        if extra:
            message.update(extra)
        await self.add_message(message)