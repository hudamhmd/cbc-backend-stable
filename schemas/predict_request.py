from pydantic import BaseModel, Field
from typing import Dict, Optional, Any

Flag = str  # LOW | NORMAL | HIGH | UNKNOWN

class PredictRequest(BaseModel):
    cbc_values: Dict[str, float] = Field(
        ...,
        description="Structured CBC numeric values. Can be canonical keys or common aliases.",
    )
    cbc_flags: Optional[Dict[str, Flag]] = Field(default=None)
    context: Optional[Dict[str, Any]] = Field(default=None)
    top_k: int = Field(default=3, ge=1, le=5)