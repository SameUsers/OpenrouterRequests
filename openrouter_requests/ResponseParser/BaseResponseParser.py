from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseResponseParser(ABC):

    @abstractmethod
    async def parse(self, response: Dict[str, Any]) -> Dict[str, Any]:
        pass