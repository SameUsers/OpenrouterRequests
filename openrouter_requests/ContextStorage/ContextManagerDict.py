from typing import Any, Dict, List, Optional
from openrouter_requests.ContextStorage.BaseContextManager import BaseContextManager


class DictContextManager(BaseContextManager):

    def __init__(self, max_messages: int = 30, default_dialog_id: str = "default") -> None:

        self._dialogs: Dict[str, List[Dict[str, Any]]] = {}
        self._max_messages = max_messages
        self._current_dialog_id = default_dialog_id
        self._dialogs.setdefault(default_dialog_id, [])

    async def set_dialog(self, dialog_id: str) -> None:
        dialog_id = str(dialog_id)
        self._current_dialog_id = dialog_id
        self._dialogs.setdefault(dialog_id, [])

    async def add_message(self, message: Dict[str, Any]) -> None:

        dialog_id = str(message.get("dialog_id") or self._current_dialog_id)
        self._current_dialog_id = dialog_id

        dialog_messages = self._dialogs.setdefault(dialog_id, [])
        dialog_messages.append(message)

        await self._trim_context(dialog_id)

    async def get_context(self, dialog_id: Optional[str] = None) -> List[Dict[str, Any]]:

        if dialog_id is None:
            dialog_id = self._current_dialog_id
        return self._dialogs.get(dialog_id, [])

    async def _trim_context(self, dialog_id: str) -> None:

        messages = self._dialogs.get(dialog_id, [])
        if len(messages) <= self._max_messages:
            return

        system_messages: List[Dict[str, Any]] = []
        non_system_messages: List[Dict[str, Any]] = []

        for msg in messages:
            if msg.get("role") == "system":
                system_messages.append(msg)
            else:
                non_system_messages.append(msg)

        overflow = len(system_messages) + len(non_system_messages) - self._max_messages
        if overflow > 0:
            non_system_messages = non_system_messages[overflow:]

        self._dialogs[dialog_id] = system_messages + non_system_messages

    async def upsert_tagged_system(self, tag: str, content: str, dialog_id: Optional[str] = None) -> None:

        if dialog_id is None:
            dialog_id = self._current_dialog_id
        messages = self._dialogs.get(dialog_id, [])
        updated = False
        for msg in messages:
            if msg.get("role") == "system" and msg.get("_tag") == tag:
                msg["content"] = content
                updated = True
                break
        if not updated:
            self._dialogs.setdefault(dialog_id, []).append({
                "role": "system",
                "content": content,
                "_tag": tag,
                "dialog_id": dialog_id,
            })
        await self._trim_context(dialog_id)


    async def reset_context(self, dialog_id: Optional[str] = None) -> None:
        if dialog_id is None:
            dialog_id = self._current_dialog_id

        if dialog_id in self._dialogs:
            del self._dialogs[dialog_id]
