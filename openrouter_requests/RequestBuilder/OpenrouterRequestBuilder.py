from typing import Any, Dict, List, Optional
from openrouter_requests.RequestBuilder.BaseRequestBuilder import BaseRequestBuilder
from openrouter_requests.schemas import OpenrouterRequest
from loguru import logger

class OpenrouterRequestBuilder(BaseRequestBuilder):

    def __init__(self) -> None:
        logger.success(
            "Инициализирован класс {} с параметрками {}",
            self.__class__.__name__,
            self.__dict__
        )

    async def build_request(
            self,
            data: OpenrouterRequest) -> Dict[str, Any]:
        return data.model_dump()