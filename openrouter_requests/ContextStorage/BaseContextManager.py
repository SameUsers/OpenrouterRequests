from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
import base64


class BaseContextManager(ABC):

    @abstractmethod
    async def add_message(self, message: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    async def get_context(self) -> List[Dict[str, Any]]:
        pass

    async def add_to_context(
            self,
            data: Union[str, List[Dict[str, Any]]],
            role: str,
            **extra: Any,
    ) -> None:
        message: Dict[str, Any] = {"role": role, "content": data}
        if extra:
            message.update(extra)
        await self.add_message(message)

    async def add_image_to_context(
            self,
            text: str,
            role: str,
            image_bytes: bytes,
            image_format: str = "png",
            **extra
    ) -> None:
        base64_str = base64.b64encode(image_bytes).decode('utf-8')
        mime_type = f"image/{image_format}"
        if image_format.lower() == "jpg":
            mime_type = "image/jpeg"

        content = [
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64_str}"
                }
            }
        ]
        await self.add_to_context(content, role, **extra)