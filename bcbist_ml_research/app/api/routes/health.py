from fastapi import APIRouter, Request
from datetime import datetime
from app.config import DATA_FEATURES_DIR

router = APIRouter()

@router.get("/")
async def get_health(request: Request):
    pm = request.app.state.production_manager
    models_status = "loaded" if pm.scorer.models else "missing"

    # Discovery latest data date
    latest_date = "UNKNOWN"
    try:
        sample_file = next(DATA_FEATURES_DIR.glob("*.csv"), None)
        if sample_file:
            with open(sample_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    latest_date = lines[-1].split(',')[0]
    except:
        pass

    return {
        "status": "healthy" if models_status == "loaded" else "degraded",
        "service": "bcbist-ml-decision-engine",
        "version": "phase22-champion",
        "timestamp": datetime.now().isoformat(),
        "model_assets": models_status,
        "latest_feature_date": latest_date
    }
