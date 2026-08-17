import sys
import os
import pandas as pd
import numpy as np
import logging
import joblib

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import DATA_REPORTS_DIR, MODELS_DIR, DATA_FEATURES_DIR, STOCK_UNIVERSE
from app.ml.meta_optimizer import MetaModelV2
from app.research.regimes import MarketRegimeDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TrainMetaV2")

def main():
    logger.info("--- PHASE 11: TRAINING META-MODEL V2 ---")

    # 1. Load Success Memory
    mem_path = DATA_REPORTS_DIR / "PHASE11_SUCCESS_MEMORY.csv"
    if not mem_path.exists():
        logger.error("Run scripts/phase11_init_memory.py first.")
        return

    mem_df = pd.read_csv(mem_path)
    mem_df = mem_df[mem_df['outcome_class'] != 'PENDING']

    # 2. Enrich with features for training
    # For a quick research demo, we'll use a subset of symbols to build the training set
    logger.info("Enriching memory with technical context...")
    enriched_data = []

    # Group memory by date for efficiency
    for date, group in mem_df.groupby('date'):
        # We need the regime for this date
        # Simulating regime from RSI
        regime = "NORMAL" # Simplified

        for _, row in group.iterrows():
            # In a real run, we'd join with the full feature row at that date/symbol
            # Here we mock the meta-features based on stored scores
            meta_features = {
                "sig_5d": row.get('raw_score', 0.5),
                "sig_1d": row.get('raw_score', 0.5) * 0.9, # Mock
                "sig_3d": row.get('raw_score', 0.5) * 0.95, # Mock
                "model_disagreement": 0.05, # Mock
                "regime_val": 0,
                "is_hit": 1 if row['actual_return_5d'] > 0 else 0
            }
            enriched_data.append(meta_features)

    train_df = pd.DataFrame(enriched_data)

    # 3. Train
    meta_v2 = MetaModelV2()
    X = train_df.drop(columns=['is_hit'])
    y = train_df['is_hit']

    meta_v2.train(X, y)

    # 4. Save
    joblib.dump(meta_v2, MODELS_DIR / "phase11_meta_v2.joblib")
    logger.info("MetaModelV2 trained and saved.")

if __name__ == "__main__":
    main()
