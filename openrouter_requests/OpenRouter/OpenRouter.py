import inspect
import json
from typing import Any, Dict, List, Type, Optional
from openrouter_requests.ContextStorage.ContextManagerLinear import LinearContextManager
from openrouter_requests.ResponseParser.OpenRouterResponseParser import OpenrouterResponseParser
from openrouter_requests.RequestBuilder.OpenrouterRequestBuilder import OpenrouterRequestBuilder
from openrouter_requests.TransportModule.BaseTransport import Transport
from openrouter_requests.TransportModule import HttpxProcessor
from openrouter_requests.ToolsModule.create_tool import Tools
from openrouter_requests.ToolsModule.tool_runner import ToolRunner
from openrouter_requests.ChromaDB.vector_base import ChromaVectorStore
from openrouter_requests.ContextStorage.BaseContextManager import BaseContextManager
from openrouter_requests.ResponseParser.BaseResponseParser import BaseResponseParser
import threading
from openrouter_requests.schemas import OpenrouterRequest
from loguru import logger

class OpenRouter:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(
            self,
            base_url: str = "https://openrouter.ai/api/v1/chat/completions",
            model: str = "deepseek/deepseek-chat-v3-0324",
            api_key: str = None,
            transport: Type[Transport] = HttpxProcessor,
            context: Type[BaseContextManager] = LinearContextManager,
            parser: Type[BaseResponseParser] = OpenrouterResponseParser,
            tool_class: Type[Tools] = ToolRunner,
            rag_store: Optional[ChromaVectorStore] = None) -> None:

        if not hasattr(self, "_initialized") or not self._initialized:
            self.model = model
            if not api_key:
                raise ValueError("Api key is None")
            self.api_key = api_key
            self.base_url = base_url
            self.request_processor: Transport = transport()
            self.context = context()
            self.builder = OpenrouterRequestBuilder()
            self.parser = parser()
            self.rag_module: Optional[ChromaVectorStore] = rag_store or ChromaVectorStore()
            self._tool_class: Type[Tools] = tool_class
            self._tool_instance: Tools = tool_class()
            self._tools_schema: List[Dict[str, Any]] | None = None
            self.header = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            self._initialized = True
            logger.success(
                "Инициализирован синглтон класса {} с параметрками {}",
                self.__class__.__name__,
                self.__dict__
            )


    async def send(
            self,
            data: str,
            role: str,
            dialog_id: Optional[str] = None,
            image: Optional[bytes] = None,
            image_format: Optional[str] = None,

    ) -> Dict[str, Any]:

        if dialog_id is not None and hasattr(self.context, "set_dialog"):
            await self.context.set_dialog(dialog_id)
        extra: Dict[str, Any] = {}
        if dialog_id is not None:
            extra["dialog_id"] = dialog_id

        if image is not None and image_format is not None:
            await self.context.add_image_to_context(
                text=data,
                role=role,
                image_bytes=image,
                image_format=image_format,
                **extra
            )
        else:
            await self.context.add_to_context(
                data=data,
                role=role,
                **extra,
            )
        if self.rag_module is not None:
            rag_docs = await self._rag_search(query=data)
            await self._add_rag_context(rag_docs, dialog_id=dialog_id)

        payload = await self.builder.build_request(data=OpenrouterRequest(
            model = self.model,
            messages = await self.context.get_context(),
            tools = await self._get_tools_schema()
        ))

        response = await self.request_processor.post(
            url=self.base_url,
            headers=self.header,
            payload=payload,
        )
        parsed = await self.parser.parse(response)
        if parsed["type"] == "message":
            await self._add_assistant_message_to_context(parsed, dialog_id=dialog_id)
            return parsed

        if parsed["type"] == "tool_calls":
            tool_results: List[Dict[str, Any]] = []
            for call in parsed["calls"]:
                result = await self._run_tool(
                    func_name=call["name"],
                    call_id=call["id"],
                    dialog_id=dialog_id,
                    **(call["arguments"] or {}),
                )
                tool_results.append(result)

            payload_followup = await self.builder.build_request(OpenrouterRequest(
                model = self.model,
                messages = await self.context.get_context(),
                tools = await self._get_tools_schema()
            ))

            response_followup = await self.request_processor.post(
                url=self.base_url,
                headers=self.header,
                payload=payload_followup,
            )

            parsed_followup = await self.parser.parse(response_followup)
            if parsed_followup["type"] == "message":
                await self._add_assistant_message_to_context(parsed_followup, dialog_id=dialog_id)
            parsed_followup["tool_results"] = tool_results
            return parsed_followup
        return parsed

    async def add_system_prompt(
            self,
            data: str,
            dialog_id: Optional[str] = None,
    ) -> None:

        extra: Dict[str, Any] = {}
        if dialog_id is not None:
            extra["dialog_id"] = dialog_id
            if hasattr(self.context, "set_dialog"):
                await self.context.set_dialog(dialog_id)

        if hasattr(self.context, "upsert_tagged_system"):
            await self.context.upsert_tagged_system(
                tag="system_prompt",
                content=data,
                dialog_id=dialog_id,
            )
        else:
            await self.context.add_to_context(data=data, role="system", **extra)

    async def _add_assistant_message_to_context(
            self,
            parsed: Dict[str, Any],
            dialog_id: Optional[str] = None,
    ) -> None:

        extra: Dict[str, Any] = {}
        if dialog_id is not None:
            extra["dialog_id"] = dialog_id

        await self.context.add_to_context(
            data=parsed["content"],
            role=parsed.get("role") or "assistant",
            **extra,
        )

    async def _get_tools_schema(self) -> List[Dict[str, Any]]:

        if self._tools_schema is None:
            self._tools_schema = await Tools.generate_tools_from_class(
                self._tool_class,
            )
        return self._tools_schema

    async def _rag_search(self, query: str) -> List[Dict[str, Any]]:

        if self.rag_module is None:
            return []

        result = await self.rag_module.search(query=query)
        docs: List[Dict[str, Any]] = []

        if not result:
            return docs

        for item in result:
            normalized = {
                "ID": item.get("id"),
                "score": item.get("score"),
                "Category": item.get("metadata", {}).get("category"),
                "Text": item.get("text"),
            }
            docs.append(normalized)

        return docs

    async def _add_rag_context(
            self,
            docs: List[Dict[str, Any]],
            dialog_id: Optional[str] = None,
    ) -> None:

        if not docs:
            return

        top_k = 5
        docs = docs[:top_k]

        lines: List[str] = []
        for idx, doc in enumerate(docs, start=1):
            category = doc.get("Category") or "unknown"
            text = doc.get("Text") or ""
            score = doc.get("score")
            lines.append(
                f"[{idx}] (category={category}, score={score})\n{text}",
            )

        rag_text = "\n\n".join(lines)
        content = f"Контекст из базы знаний (RAG-поиск):\n{rag_text}"

        extra: Dict[str, Any] = {}
        if dialog_id is not None:
            extra["dialog_id"] = dialog_id

        if hasattr(self.context, "upsert_tagged_system"):
            await self.context.upsert_tagged_system(
                tag="rag_context",
                content=content,
                **extra,
            )
        else:
            await self.context.add_to_context(
                data=content,
                role="system",
                **extra,
            )

    async def _run_tool(
            self,
            func_name: str,
            call_id: str,
            dialog_id: Optional[str] = None,
            **kwargs: Any,
    ) -> Dict[str, Any]:

        method = getattr(self._tool_instance, func_name, None)
        if method is None or not callable(method):
            raise ValueError(f"Метод инструмента '{func_name}' не реализован")

        result = method(**kwargs)
        if inspect.iscoroutine(result):
            result = await result

        content_str = (
            result
            if isinstance(result, str)
            else json.dumps(result, ensure_ascii=False)
        )

        tool_message: Dict[str, Any] = {
            "tool_call_id": call_id,
            "role": "tool",
            "name": func_name,
            "content": content_str,
        }

        extra: Dict[str, Any] = {"tool_call_id" : tool_message["tool_call_id"]}
        if dialog_id is not None:
            extra["dialog_id"] = dialog_id

        await self.context.add_to_context(
            data=tool_message["content"],
            role=tool_message["role"],
            **extra,
        )

        return tool_message