from pydantic import BaseModel, Field
from typing import Any

class OpenrouterRequest(BaseModel):
    model: str = Field(...)
    messages: list[dict[str,str]]
    tools: list[dict[str, Any]] | None = None