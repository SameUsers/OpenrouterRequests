import json
from typing import Any, Dict, List
from openrouter_requests.ResponseParser.BaseResponseParser import BaseResponseParser
from loguru import logger

class OpenrouterResponseParser(BaseResponseParser):

    def __init__(self) -> None:
        logger.success(
            "Инициализирован класс {} с параметрками {}",
            self.__class__.__name__,
            self.__dict__
        )

    async def parse(self, response: Dict[str, Any]) -> Dict[str, Any]:

        choices = response.get("choices", [])
        if not choices:
            return {
                "type": "empty",
                "role": None,
                "content": "",
                "calls": [],
            }

        message = choices[0].get("message", {}) or {}
        role = message.get("role")
        content = message.get("content") or ""
        tool_calls = message.get("tool_calls") or []

        parsed_calls: List[Dict[str, Any]] = []

        for call in tool_calls:
            function_block = call.get("function", {}) or {}
            name = function_block.get("name")
            raw_args = function_block.get("arguments", "{}")

            try:
                args = json.loads(raw_args)
            except Exception:
                args = {"_raw": raw_args}

            parsed_calls.append(
                {
                    "id": call.get("id"),
                    "name": name,
                    "arguments": args,
                }
            )

        if parsed_calls:
            return {
                "type": "tool_calls",
                "role": role,
                "content": content,
                "calls": parsed_calls,
            }

        return {
            "type": "message",
            "role": role,
            "content": content,
            "calls": [],
        }