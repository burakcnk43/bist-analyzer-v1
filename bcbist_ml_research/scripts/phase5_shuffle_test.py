import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import STOCK_UNIVERSE, DATA_FEATURES_DIR
from app.ml.pipeline import ResearchPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ShuffleTest")

def main():
    logger.info("--- PHASE 5: NEGATIVE CONTROL (SHUFFLE TEST) ---")

    # 1. Load Data
    all_data_list = []
    for entry in STOCK_UNIVERSE:
        symbol = entry['symbol']
        path = DATA_FEATURES_DIR / f"{symbol.replace('.', '_')}_features.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, index_col='date', parse_dates=True)
            df['symbol_col'] = symbol
            all_data_list.append(df)
    global_df = pd.concat(all_data_list).sort_index()

    holdout_start = global_df.index.max() - timedelta(days=180)
    research_df = global_df[global_df.index < holdout_start]
    holdout_df = global_df[global_df.index >= holdout_start]

    h = 20
    target_cls = f'target_up_{h}d'

    # Standard feature set
    feature_patterns = ['rsi_', 'sma_', 'macd', 'bb_', 'dist_', 'atr_', 'volume', 'obv', 'rel_sector', 'sector_mean', 'macro_', 'event_']
    drop_cols = ['open', 'high', 'low', 'close', 'volume', 'adj_close', 'symbol_col', 'sector_col']
    selected_features = sorted(list(set([c for c in global_df.columns if any(p in c for p in feature_patterns)
                                       and c not in drop_cols and not c.startswith('target_')])))

    # --- EXPERIMENT 1: NORMAL MODEL ---
    logger.info("Running Normal Model...")
    p1 = ResearchPipeline(model_type='xgboost', n_features=30)
    tr1 = research_df[selected_features + [target_cls]].dropna()
    p1.fit(tr1[selected_features], tr1[target_cls])
    te1 = holdout_df[selected_features + [target_cls]].dropna()
    acc1 = accuracy_score(te1[target_cls], p1.predict(te1[selected_features]))
    logger.info(f"Normal Model Accuracy: {acc1:.4%}")

    # --- EXPERIMENT 2: SHUFFLED TARGET ---
    logger.info("Running Shuffled-Target Model (Signal Destroyed)...")
    p2 = ResearchPipeline(model_type='xgboost', n_features=30)
    tr2 = tr1.copy()
    # Shuffle targets IN-PLACE for training
    tr2[target_cls] = np.random.permutation(tr2[target_cls].values)
    p2.fit(tr2[selected_features], tr2[target_cls])

    # Evaluate on the SAME real holdout
    acc2 = accuracy_score(te1[target_cls], p2.predict(te1[selected_features]))
    logger.info(f"Shuffled-Target Model Accuracy: {acc2:.4%}")

    # --- EXPERIMENT 3: SHUFFLED FEATURES ---
    logger.info("Running Shuffled-Features Model...")
    p3 = ResearchPipeline(model_type='xgboost', n_features=30)
    tr3 = tr1.copy()
    for col in selected_features:
        tr3[col] = np.random.permutation(tr3[col].values)
    p3.fit(tr3[selected_features], tr3[target_cls])
    acc3 = accuracy_score(te1[target_cls], p3.predict(te1[selected_features]))
    logger.info(f"Shuffled-Features Model Accuracy: {acc3:.4%}")

    print("\n" + "="*40)
    print("SHUFFLE TEST RESULTS")
    print("="*40)
    print(f"Normal:   {acc1:.2%}")
    print(f"Shuffled Target: {acc2:.2%}")
    print(f"Shuffled Feats:  {acc3:.2%}")
    print("="*40)

    if acc2 > 0.55 or acc3 > 0.55:
         print("!! WARNING: Shuffled models still perform well. POSSIBLE LEAKAGE DETECTED !!")
    else:
         print("PASS: Signal destroyed as expected.")

if __name__ == "__main__":
    main()
