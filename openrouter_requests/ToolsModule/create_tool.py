import inspect
from typing import List, Dict, Any, Type, Optional, get_type_hints, Union
from abc import ABC


class Tools(ABC):
    @staticmethod
    async def generate_tools_from_class(
            cls: Type["Tools"],
    ) -> List[Dict[str, Any]]:
        if not issubclass(cls, Tools):
            raise TypeError(f"Класс {cls.__name__} должен наследовать от Tools")

        tools: List[Dict[str, Any]] = []

        for name, func in inspect.getmembers(cls, predicate=inspect.isfunction):
            if name.startswith("_"):
                continue

            qual_name_parts = func.__qualname__.split(".")
            if len(qual_name_parts) < 2 or qual_name_parts[0] != cls.__name__:
                continue

            description, param_descriptions = Tools._parse_docstring(func.__doc__)

            signature = inspect.signature(func)
            type_hints = get_type_hints(func)

            properties: Dict[str, Any] = {}
            required: List[str] = []

            for param_name, param in signature.parameters.items():
                if param_name == "self":
                    continue

                ann = type_hints.get(param_name, str)
                param_schema = Tools._python_type_to_schema(ann)

                if param_name in param_descriptions:
                    param_schema["description"] = param_descriptions[param_name]

                properties[param_name] = param_schema

                has_default = param.default is not inspect.Parameter.empty
                if not has_default and not Tools._is_optional_type(ann):
                    required.append(param_name)

            tool = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                        "additionalProperties": False
                    },
                },
            }
            tools.append(tool)
        return tools

    @staticmethod
    def _parse_docstring(docstring: Optional[str]) -> tuple[str, Dict[str, str]]:
        if not docstring:
            return "Описание отсутствует", {}

        lines = docstring.strip().split('\n')
        description_lines = []
        param_lines_started = False
        param_descriptions = {}

        for line in lines:
            line = line.strip()

            if line.lower().startswith("параметры:") or line.lower().startswith("params:"):
                param_lines_started = True
                continue

            if param_lines_started:
                if line.startswith("-") or ":" in line:
                    clean_line = line.lstrip('- ').strip()
                    if ":" in clean_line:
                        param_name, param_desc = clean_line.split(":", 1)
                        param_descriptions[param_name.strip()] = param_desc.strip()
            else:
                if line:
                    description_lines.append(line)

        description = " ".join(description_lines) if description_lines else "Описание отсутствует"

        return description, param_descriptions

    @staticmethod
    def _python_type_to_schema(python_type) -> Dict[str, Any]:
        origin = getattr(python_type, "__origin__", None)
        if origin is Optional or (origin is Union and type(None) in python_type.__args__):
            args = python_type.__args__
            real_type = next(arg for arg in args if arg is not type(None))
            schema = Tools._python_type_to_schema(real_type)
            return schema

        type_mapping = {
            str: {"type": "string"},
            int: {"type": "integer"},
            float: {"type": "number"},
            bool: {"type": "boolean"},
            list: {"type": "array"},
            dict: {"type": "object"},
        }

        if python_type in type_mapping:
            return type_mapping[python_type]

        return {"type": "string"}

    @staticmethod
    def _is_optional_type(python_type) -> bool:
        origin = getattr(python_type, "__origin__", None)
        if origin is Optional:
            return True
        if origin is Union:
            return type(None) in python_type.__args__
        return False