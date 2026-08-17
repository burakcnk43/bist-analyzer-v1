import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import joblib

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import STOCK_UNIVERSE, DATA_FEATURES_DIR, MODELS_DIR
from app.ml.pipeline import ResearchPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MultiHorizonTraining")

def main():
    logger.info("--- PHASE 10: TRAINING SPECIALIZED MODELS (ALIGNED WINDOWS) ---")

    all_data_list = []
    for entry in STOCK_UNIVERSE:
        symbol = entry['symbol']
        path = DATA_FEATURES_DIR / f"{symbol.replace('.', '_')}_features.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, index_col='date', parse_dates=True)
            df['symbol_col'] = symbol
            all_data_list.append(df)
    global_df = pd.concat(all_data_list).sort_index()

    # Align with Phase 9 Research Window
    # Unseen period starts 60 days ago. Purge 20 days.
    max_date = global_df.index.max()
    train_end = max_date - timedelta(days=60 + 20)

    train_df = global_df[global_df.index < train_end]

    horizons = [1, 3, 5]
    feature_patterns = ['rsi_', 'sma_', 'macd', 'bb_', 'dist_', 'atr_', 'volume', 'obv', 'rel_sector', 'sector_mean', 'macro_', 'event_']
    selected_features = sorted(list(set([c for c in global_df.columns if any(p in c for p in feature_patterns)
                                       and not any(x in c for x in ['target', 'symbol', 'sector', 'col'])])))

    for h in horizons:
        logger.info(f"Training specialized models for Horizon: {h}d (End Date: {train_end.date()})")

        target_cls = f'target_up_{h}d'
        target_rnk = f'target_label_{h}d'

        p_cls = ResearchPipeline(model_type='xgboost', n_features=30, task='classification')
        tr_c = train_df[selected_features + [target_cls]].dropna()
        p_cls.fit(tr_c[selected_features], tr_c[target_cls])

        p_rnk = ResearchPipeline(model_type='xgboost', n_features=30, task='ranking')
        tr_r = train_df[selected_features + [target_rnk]].dropna()
        p_rnk.fit(tr_r[selected_features], tr_r[target_rnk])

        joblib.dump(p_cls, MODELS_DIR / f"phase10_cls_{h}d.joblib")
        joblib.dump(p_rnk, MODELS_DIR / f"phase10_rnk_{h}d.joblib")

    logger.info("Training complete.")

if __name__ == "__main__":
    main()
