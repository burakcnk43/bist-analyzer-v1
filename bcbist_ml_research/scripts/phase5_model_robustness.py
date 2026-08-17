import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import STOCK_UNIVERSE, DATA_FEATURES_DIR, DATA_REPORTS_DIR
from app.ml.pipeline import ResearchPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ModelRobustness")

def main():
    logger.info("--- PHASE 5: MODEL & SEED ROBUSTNESS AUDIT ---")

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
    feature_patterns = ['rsi_', 'sma_', 'macd', 'bb_', 'dist_', 'atr_', 'volume', 'obv', 'rel_sector', 'sector_mean', 'macro_', 'event_']
    drop_cols = ['open', 'high', 'low', 'close', 'volume', 'adj_close', 'symbol_col', 'sector_col']
    selected_features = sorted(list(set([c for c in global_df.columns if any(p in c for p in feature_patterns)
                                       and c not in drop_cols and not c.startswith('target_')])))

    train_clean = research_df[selected_features + [target_cls]].dropna()
    test_clean = holdout_df[selected_features + [target_cls]].dropna()

    # --- EXPERIMENT 1: MODEL TYPE ROBUSTNESS ---
    model_types = ['xgboost', 'rf', 'logistic']
    model_results = []

    for mt in model_types:
        logger.info(f"Testing Model: {mt}")
        p = ResearchPipeline(model_type=mt, n_features=30)
        p.fit(train_clean[selected_features], train_clean[target_cls])
        acc = accuracy_score(test_clean[target_cls], p.predict(test_clean[selected_features]))
        model_results.append({"Model": mt, "Accuracy": acc})

    # --- EXPERIMENT 2: SEED ROBUSTNESS (XGBoost) ---
    seeds = [1, 7, 21, 42, 100]
    seed_results = []
    for s in seeds:
        logger.info(f"Testing Seed: {s}")
        # Need to modify Pipeline to accept seed, but let's just use xgb params
        p = ResearchPipeline(model_type='xgboost', n_features=30)
        p.model.params['random_state'] = s
        p.fit(train_clean[selected_features], train_clean[target_cls])
        acc = accuracy_score(test_clean[target_cls], p.predict(test_clean[selected_features]))
        seed_results.append({"Seed": s, "Accuracy": acc})

    print("\nModel Type Results:")
    print(pd.DataFrame(model_results).to_markdown(index=False))

    print("\nSeed Results (XGBoost):")
    print(pd.DataFrame(seed_results).to_markdown(index=False))

    pd.DataFrame(model_results).to_csv(DATA_REPORTS_DIR / "PHASE5_MODEL_ROBUSTNESS.csv", index=False)
    pd.DataFrame(seed_results).to_csv(DATA_REPORTS_DIR / "PHASE5_SEED_ROBUSTNESS.csv", index=False)

if __name__ == "__main__":
    main()
