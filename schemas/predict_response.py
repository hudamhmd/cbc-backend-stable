from pydantic import BaseModel
from typing import Dict, List, Any

class PredictResponse(BaseModel):
    stage1: Dict[str, Any]
    path: str  # "CBC" or "NON_CBC"
    top_predictions: List[Dict[str, Any]]
    ontology_support: List[Dict[str, Any]]
    urgent_attention: bool
    recommended_tests: List[Dict[str, Any]]
    specialty: List[str]
    red_flags: List[str]
    warnings: List[str]
    disclaimer: str