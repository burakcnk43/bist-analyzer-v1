from fastapi import FastAPI
from app.api.routes import health, research, models, sectors, predictions
from app.ml.production_manager import ProductionManager
from app.config import BASE_DIR

app = FastAPI(
    title="BCBIST ML Decision Engine",
    description="Production-grade stock selection engine for BIST",
    version="1.0.0"
)

# Initialize Champion Decision Engine
app.state.production_manager = ProductionManager(
    BASE_DIR / "configs" / "production" / "production_config.json"
)

app.include_router(health.router, tags=["Health"])
app.include_router(research.router, prefix="/api/research", tags=["Research"])
app.include_router(models.router, prefix="/api/research/models", tags=["Models"])
app.include_router(sectors.router, prefix="/api/research/sectors", tags=["Sectors"])
app.include_router(predictions.router, prefix="/api/research/predictions", tags=["Predictions"])

@app.get("/")
async def root():
    return {"message": "Welcome to BCBIST ML Research Engine API"}
