import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import STOCK_UNIVERSE, DATA_FEATURES_DIR, DATA_REPORTS_DIR
from app.ml.pipeline import ResearchPipeline
from app.research.ranking_metrics import calculate_top_k_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ModelComparison")

def main():
    logger.info("--- PHASE 12: MODEL FAMILY COMPARISON (XGB vs LGBM vs CatBoost) ---")

    # 1. Load Data
    all_data_list = []
    valid_syms = pd.read_csv('data/raw/bist_universe_valid.csv')['symbol'].tolist()
    for symbol in valid_syms[:50]:
        path = DATA_FEATURES_DIR / f"{symbol.replace('.', '_')}_features.csv"
        if path.exists():
            df = pd.read_csv(path, index_col='date', parse_dates=True)
            all_data_list.append(df)
    global_df = pd.concat(all_data_list).sort_index()

    test_start = global_df.index.max() - timedelta(days=90)
    train_df = global_df[global_df.index < (test_start - timedelta(days=20))]
    test_df = global_df[global_df.index >= test_start]

    feature_patterns = ['rsi_', 'sma_', 'macd', 'bb_', 'dist_', 'atr_', 'volume']
    selected_features = [c for c in global_df.columns if any(p in c for p in feature_patterns)]
    target = 'target_label_5d'
    ret_target = 'target_return_5d'

    families = ['xgboost', 'lightgbm'] # Catboost often needs separate install/heavy
    comparison = []

    for family in families:
        logger.info(f"Evaluating model family: {family}...")
        try:
            model = ResearchPipeline(model_type=family, n_features=30, task='ranking')
            model.fit(train_df[selected_features + [target]].dropna(), train_df[target].dropna())

            test_subset = test_df[selected_features + [ret_target]].dropna().copy()
            test_subset['alpha'] = model.predict(test_subset[selected_features])

            summary, _ = calculate_top_k_metrics(test_subset, 'alpha', ret_target, k=5)
            comparison.append({
                "Family": family,
                "Hit_Rate@5": summary['avg_hit_rate'],
                "Avg_Return": summary['avg_return']
            })
        except Exception as e:
            logger.error(f"Error evaluating {family}: {e}")

    res_df = pd.DataFrame(comparison)
    res_df.to_csv(DATA_REPORTS_DIR / "PHASE12_MODEL_COMPARISON.csv", index=False)
    print("\nModel Family Comparison:")
    print(res_df.to_markdown(index=False))

if __name__ == "__main__":
    main()
