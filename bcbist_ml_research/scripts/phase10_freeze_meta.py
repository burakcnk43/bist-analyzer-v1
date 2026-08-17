import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import joblib

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import STOCK_UNIVERSE, DATA_FEATURES_DIR, DATA_REPORTS_DIR, MODELS_DIR
from app.ml.meta_models import SuccessPredictor
from app.features.similarity import SimilarityEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FreezeMeta")

def main():
    logger.info("--- PHASE 10: FREEZING META-LEARNING COMPONENTS (UP TO AUG 4) ---")

    # 1. Load Research Data (for Similarity Engine)
    all_data_list = []
    # Use top 150 for similarity index to keep search fast and relevant
    valid_syms = pd.read_csv('data/raw/bist_universe_valid.csv')['symbol'].tolist()
    for symbol in valid_syms:
        path = DATA_FEATURES_DIR / f"{symbol.replace('.', '_')}_features.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, index_col='date', parse_dates=True)
            # STRICY PIT: Only data up to Aug 4
            df = df[df.index <= '2026-08-04']
            if not df.empty:
                df['symbol_col'] = symbol
                all_data_list.append(df)

    global_df = pd.concat(all_data_list).sort_index()

    # 2. Train and Save SuccessPredictor
    log_path = DATA_REPORTS_DIR / "PHASE8_PREDICTION_LOG.csv"
    log_df = pd.read_csv(log_path)
    log_df['date'] = pd.to_datetime(log_df['date'])
    # PIT: Only logs up to Aug 4
    log_df = log_df[log_df['date'] <= '2026-08-04']

    # Identify available features (same as research scripts)
    feature_patterns = ['rsi_', 'sma_', 'macd', 'bb_', 'dist_', 'atr_', 'volume', 'obv', 'rel_sector', 'sector_mean', 'macro_', 'event_']
    selected_features = sorted(list(set([c for c in global_df.columns if any(p in c for p in feature_patterns)
                                       and not any(x in c for x in ['target', 'symbol', 'sector', 'col'])])))

    logger.info(f"Training SuccessPredictor on {len(log_df)} OOS rows...")
    success_mdl = SuccessPredictor()
    success_mdl.train(log_df, selected_features)
    joblib.dump(success_mdl, MODELS_DIR / "phase10_success_predictor.joblib")

    # 3. Fit and Save SimilarityEngine
    logger.info(f"Fitting SimilarityEngine on {len(global_df)} rows...")
    sim_engine = SimilarityEngine(k=20)
    sim_engine.fit(global_df.dropna(subset=['target_return_5d']), 'target_return_5d')
    joblib.dump(sim_engine, MODELS_DIR / "phase10_similarity_engine.joblib")

    logger.info("Meta-models frozen and saved to models/ directory.")

if __name__ == "__main__":
    main()
