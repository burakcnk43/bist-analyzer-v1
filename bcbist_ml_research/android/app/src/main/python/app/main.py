import asyncio
import logging
from datetime import datetime, time
import pandas as pd
from fastapi import FastAPI
from app.api.routes import health, research, models, sectors, predictions
from app.ml.production_manager import ProductionManager
from app.config import BASE_DIR, BREADTH_DATA_PATH, DATA_FEATURES_DIR

# Set up logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BCBIST ML Decision Engine",
    description="Production-grade stock selection engine for BIST",
    version="1.0.0"
)

# Initialize Champion Decision Engine
try:
    app.state.production_manager = ProductionManager(
        BASE_DIR / "configs" / "production" / "production_config.json"
    )
    logger.info("Champion Decision Engine initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Decision Engine: {e}")
    # In production, we might want to crash or start in degraded mode.
    # For now, we'll let it crash so Railway logs show the error.
    raise e

async def daily_report_scheduler():
    """
    Background task that triggers PDF generation at 10:30 AM on trading days.
    """
    while True:
        try:
            now = datetime.now()
            target_time = time(10, 30)

            if now.weekday() < 5 and now.time() >= target_time and now.hour == 10 and now.minute <= 35:
                logger.info("Auto-triggering 10:30 AM report generation...")
                if BREADTH_DATA_PATH.exists():
                    mkt_df = pd.read_csv(BREADTH_DATA_PATH, index_col='date', parse_dates=True)
                    latest_date = mkt_df.index.max()

                    all_dfs = []
                    for f in DATA_FEATURES_DIR.glob("*.csv"):
                        df = pd.read_csv(f, index_col='date', parse_dates=True)
                        if latest_date in df.index:
                            row = df.loc[[latest_date]].copy()
                            row['symbol_col'] = f.stem.replace('_features', '').replace('_', '.')
                            all_dfs.append(row)

                    if all_dfs:
                        day_data = pd.concat(all_dfs)
                        mkt_data = mkt_df[mkt_df.index <= latest_date].tail(20)
                        app.state.production_manager.get_daily_picks(day_data, mkt_data)
                        logger.info(f"Scheduled report for {latest_date} generated.")
        except Exception as e:
            logger.error(f"Scheduled generation error: {e}")

        await asyncio.sleep(600)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(daily_report_scheduler())

# Include routers
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(research.router, prefix="/api/research", tags=["Research"])
app.include_router(models.router, prefix="/api/research/models", tags=["Models"])
app.include_router(sectors.router, prefix="/api/research/sectors", tags=["Sectors"])
app.include_router(predictions.router, prefix="/api/research/predictions", tags=["Predictions"])

@app.get("/")
async def root():
    return {
        "message": "Welcome to BCBIST ML Decision Engine API",
        "status": "online",
        "endpoints": {
            "health": "/health",
            "daily_picks": "/api/research/predictions/daily-picks",
            "pdf_report": "/api/research/predictions/report-pdf"
        }
    }
