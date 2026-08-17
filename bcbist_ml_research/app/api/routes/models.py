from fastapi import APIRouter, Request
from app.config import MODELS_DIR

router = APIRouter()

@router.get("/status")
async def get_models_status(request: Request):
    pm = request.app.state.production_manager

    # Check core multi-horizon models
    core_models = {h: h in pm.scorer.models for h in [1, 3, 5]}

    # Check specialists
    specialists = {
        "transition_expert": pm.transition_expert.is_trained,
        "group_predictor": hasattr(pm.group_predictor, 'models'),
        "alpha_trust": hasattr(pm.trust_model, 'model'),
        "darvas_quality": hasattr(pm.darvas_scorer, 'model'),
        "event_chain": hasattr(pm.event_expert, 'model')
    }

    return {
        "core_ensemble_horizons": core_models,
        "specialists_loaded": specialists,
        "models_dir": str(MODELS_DIR)
    }

@router.get("/")
async def get_models():
    return {"models": []}

@router.get("/{model_id}")
async def get_model(model_id: str):
    return {"model_id": model_id, "status": "Not found"}
