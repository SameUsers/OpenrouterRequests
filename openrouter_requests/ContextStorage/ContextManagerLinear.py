from typing import Any, Dict, List

from openrouter_requests.ContextStorage.BaseContextManager import BaseContextManager


class LinearContextManager(BaseContextManager):

    def __init__(self, max_messages: int = 30) -> None:

        self._context: List[Dict[str, Any]] = []
        self._max_messages = max_messages

    async def add_message(self, message: Dict[str, Any]) -> None:

        self._context.append(message)
        await self._trim_context()

    async def upsert_tagged_system(self, tag: str, content: str) -> None:

        updated = False

        for msg in self._context:
            if msg.get("role") == "system" and msg.get("_tag") == tag:
                msg["content"] = content
                updated = True
                break

        if not updated:
            self._context.append(
                {
                    "role": "system",
                    "content": content,
                    "_tag": tag,
                },
            )

        await self._trim_context()

    async def get_context(self) -> List[Dict[str, Any]]:

        return self._context

    async def _trim_context(self) -> None:

        if len(self._context) <= self._max_messages:
            return

        system_messages: List[Dict[str, Any]] = []
        non_system_messages: List[Dict[str, Any]] = []

        for msg in self._context:
            if msg.get("role") == "system":
                system_messages.append(msg)
            else:
                non_system_messages.append(msg)

        overflow = len(system_messages) + len(non_system_messages) - self._max_messages
        if overflow > 0:
            non_system_messages = non_system_messages[overflow:]

        self._context = system_messages + non_system_messages

    async def reset(self):
        self._context=[]