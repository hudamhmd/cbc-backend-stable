from fastapi import APIRouter
from app import ARTIFACTS

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok", "loaded": list(ARTIFACTS.keys())}