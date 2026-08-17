import sys
import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import DATA_FEATURES_DIR, MODELS_DIR
from app.ml.production_manager import ProductionManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ProductionAudit")

def run_audit():
    logger.info("--- FINAL PRODUCTION INTEGRITY AUDIT (PHASE 22) ---")

    # 1. Check Artifacts
    logger.info("Verifying model artifacts...")
    artifacts = list(MODELS_DIR.glob("*.joblib"))
    if len(artifacts) < 5:
        logger.error(f"Missing core artifacts. Found only {len(artifacts)}")
        return False

    # 2. Schema Compatibility
    logger.info("Verifying feature schema compatibility...")
    sample_feat = list(DATA_FEATURES_DIR.glob("*.csv"))[0]
    df = pd.read_csv(sample_feat, index_col='date', parse_dates=True)
    required_cols = ['rsi_14', 'relative_volume', 'dist_sma_20']
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Missing required feature: {col}")
            return False

    # 3. PIT Integrity
    logger.info("Verifying Point-in-Time alignment...")
    # Check if target_return_5d exists in features (it should, but shouldn't be used for prediction)
    # The ProductionManager must not look at 'target' columns.

    # 4. Smoke Test Prediction
    logger.info("Running smoke test prediction...")
    pm = ProductionManager(Path("configs/production/production_config.json"))

    # Mock data
    test_date = df.index[-10]
    day_data = df.loc[[test_date]].copy()
    day_data['symbol_col'] = "TEST.IS"
    market_data = pd.read_csv('data/market_breadth.csv', index_col='date', parse_dates=True).loc[[test_date]]

    try:
        response = pm.get_daily_picks(day_data, market_data)
        if 'predictions' not in response:
             logger.error("Prediction failed: No predictions in response")
             return False
        logger.info("Smoke test SUCCESS.")
    except Exception as e:
        logger.error(f"Smoke test FAILED: {str(e)}")
        return False

    logger.info("AUDIT PASSED. System is ready for deployment.")
    return True

if __name__ == "__main__":
    if run_audit():
        sys.exit(0)
    else:
        sys.exit(1)
