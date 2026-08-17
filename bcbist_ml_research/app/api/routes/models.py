from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def get_models():
    return {"models": []}

@router.get("/{model_id}")
async def get_model(model_id: str):
    return {"model_id": model_id, "status": "Not found"}
