from fastapi import APIRouter
from datetime import datetime

router = APIRouter()

@router.get("/")
async def get_health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "BCBIST ML Research Engine"
    }
