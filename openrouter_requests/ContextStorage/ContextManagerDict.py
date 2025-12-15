from typing import Any, Dict, List, Optional
from openrouter_requests.ContextStorage.BaseContextManager import BaseContextManager
import threading
import asyncio


class DictContextManager(BaseContextManager):
    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, max_messages: int = 30, default_dialog_id: str = "default") -> None:
        if self._initialized:
            return

        self._dialogs: Dict[str, List[Dict[str, Any]]] = {}
        self._max_messages = max_messages
        self._current_dialog_id = default_dialog_id
        self._dialogs.setdefault(default_dialog_id, [])
        self._dialog_locks: Dict[str, asyncio.Lock] = {}
        self._meta_lock = asyncio.Lock()
        self._initialized = True

    async def _get_dialog_lock(self, dialog_id: str) -> asyncio.Lock:
        async with self._meta_lock:
            if dialog_id not in self._dialog_locks:
                self._dialog_locks[dialog_id] = asyncio.Lock()
            return self._dialog_locks[dialog_id]

    async def set_dialog(self, dialog_id: str) -> None:
        dialog_id = str(dialog_id)

        lock = await self._get_dialog_lock(dialog_id)
        async with lock:
            self._current_dialog_id = dialog_id
            self._dialogs.setdefault(dialog_id, [])

    async def add_message(self, message: Dict[str, Any]) -> None:
        dialog_id = str(message.get("dialog_id") or self._current_dialog_id)

        lock = await self._get_dialog_lock(dialog_id)
        async with lock:
            self._current_dialog_id = dialog_id
            dialog_messages = self._dialogs.setdefault(dialog_id, [])
            dialog_messages.append(message)
            self._trim_context_sync(dialog_id)

    async def get_context(self, dialog_id: Optional[str] = None) -> List[Dict[str, Any]]:
        did = dialog_id or self._current_dialog_id

        lock = await self._get_dialog_lock(did)
        async with lock:
            return list(self._dialogs.get(did, []))

    async def upsert_tagged_system(
        self,
        tag: str,
        content: str,
        dialog_id: Optional[str] = None,
    ) -> None:
        did = dialog_id or self._current_dialog_id

        lock = await self._get_dialog_lock(did)
        async with lock:
            messages = self._dialogs.setdefault(did, [])

            for msg in messages:
                if msg.get("role") == "system" and msg.get("_tag") == tag:
                    msg["content"] = content
                    break
            else:
                messages.append({
                    "role": "system",
                    "content": content,
                    "_tag": tag,
                    "dialog_id": did,
                })

            self._trim_context_sync(did)

    async def reset_context(self, dialog_id: Optional[str] = None) -> None:
        did = dialog_id or self._current_dialog_id
        async with self._meta_lock:
            self._dialogs.pop(did, None)
            self._dialog_locks.pop(did, None)


    def _trim_context_sync(self, dialog_id: str) -> None:
        messages = self._dialogs.get(dialog_id)
        if not messages or len(messages) <= self._max_messages:
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