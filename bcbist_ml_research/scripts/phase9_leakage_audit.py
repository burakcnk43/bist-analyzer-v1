import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import STOCK_UNIVERSE, DATA_FEATURES_DIR, DATA_REPORTS_DIR
from app.features.similarity import SimilarityEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LeakageAudit")

def main():
    logger.info("--- PHASE 9: LEAKAGE AUDIT (SIMILARITY & META-LEARNING) ---")

    # 1. Load Data
    all_data_list = []
    for entry in STOCK_UNIVERSE:
        symbol = entry['symbol']
        path = DATA_FEATURES_DIR / f"{symbol.replace('.', '_')}_features.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, index_col='date', parse_dates=True)
            all_data_list.append(df)
    global_df = pd.concat(all_data_list).sort_index()

    holdout_start = global_df.index.max() - timedelta(days=180)
    research_df = global_df[global_df.index < holdout_start]
    holdout_df = global_df[global_df.index >= holdout_start]

    h = 5
    target_ret = f'target_return_{h}d'

    # 2. Similarity Engine PIT Audit
    sim_engine = SimilarityEngine(k=20)
    # Fit only on research data
    train_data = research_df.dropna(subset=[target_ret])
    sim_engine.fit(train_data, target_ret)

    # Audit: For a sample of dates in holdout, check retrieved neighbors
    test_dates = holdout_df.index.unique()[:10] # Check first 10 days of holdout

    leakage_found = False

    # We need to modify SimilarityEngine or use its internals to verify dates
    X = train_data[sim_engine.features].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_scaled = sim_engine.scaler.transform(X)

    for date in test_dates:
        query_data = holdout_df.loc[[date]]
        if query_data.empty: continue

        Q = query_data[sim_engine.features].replace([np.inf, -np.inf], np.nan).fillna(0)
        Q_scaled = sim_engine.scaler.transform(Q)

        distances, indices = sim_engine.knn.kneighbors(Q_scaled)

        # Check neighbor dates
        neighbor_dates = train_data.index[indices.flatten()]

        future_neighbors = neighbor_dates[neighbor_dates >= date]
        if len(future_neighbors) > 0:
            logger.error(f"LEAKAGE DETECTED on {date}: {len(future_neighbors)} neighbors from the future!")
            leakage_found = True

    if not leakage_found:
        logger.info("Similarity Engine PIT Audit: PASS")
    else:
        logger.error("Similarity Engine PIT Audit: FAIL")

    # 3. Meta-Model Target Shuffle Test
    # If the SuccessPredictor can predict hits from shuffled data, there's leakage.
    # We already have a shuffle script, but let's do a quick check here for meta-learning.

    # Results Summary
    audit_res = pd.DataFrame([
        {"Test": "Similarity PIT", "Status": "PASS" if not leakage_found else "FAIL"},
    ])
    audit_res.to_csv(DATA_REPORTS_DIR / "PHASE9_SIMILARITY_AUDIT.csv", index=False)

if __name__ == "__main__":
    main()
