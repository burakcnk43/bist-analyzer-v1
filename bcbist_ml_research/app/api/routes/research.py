from fastapi import APIRouter

router = APIRouter()

@router.get("/status")
async def get_status():
    return {"status": "Research engine idle"}

@router.get("/summary")
async def get_summary():
    return {"summary": "No research history found"}

@router.get("/report")
async def get_report():
    return {"message": "No reports generated yet"}
