from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

class OpenrouterRequest(BaseModel):
    model: str = Field(...)
    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]] = None