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
logger = logging.getLogger("RollingHoldouts")

def main():
    logger.info("--- PHASE 5: ROLLING HOLDOUT ROBUSTNESS ---")

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

    # Define the "Phase 4 Holdout" boundary
    final_holdout_start = global_df.index.max() - timedelta(days=180)

    # Configuration
    h = 20
    target_cls = f'target_up_{h}d'
    # Use features from the frozen config if possible, or just the group logic
    feature_patterns = ['rsi_', 'sma_', 'macd', 'bb_', 'dist_', 'atr_', 'volume', 'obv', 'rel_sector', 'sector_mean', 'macro_', 'event_']
    drop_cols = ['open', 'high', 'low', 'close', 'volume', 'adj_close', 'symbol_col', 'sector_col']
    all_target_cols = [c for c in global_df.columns if c.startswith('target_')]

    selected_features = sorted(list(set([c for c in global_df.columns if any(p in c for p in feature_patterns)
                                       and c not in drop_cols and c not in all_target_cols])))

    # Rolling Windows
    # We create 4 windows of 120 days each, moving backwards from the final holdout start
    windows = []
    curr_end = final_holdout_start
    for i in range(4):
        win_start = curr_end - timedelta(days=120)
        windows.append((win_start, curr_end))
        curr_end = win_start

    results = []

    for win_idx, (start, end) in enumerate(windows):
        win_name = f"Window {chr(65+win_idx)}"
        logger.info(f"Evaluating {win_name}: {start.date()} to {end.date()}")

        # Data split
        # Train on EVERYTHING before start (purged by 20 days)
        train_end = start - timedelta(days=20)
        train_df = global_df[global_df.index < train_end]
        test_df = global_df[(global_df.index >= start) & (global_df.index < end)]

        if train_df.empty or test_df.empty:
            logger.warning(f"Skipping {win_name} due to insufficient data.")
            continue

        # Fold-safe training
        pipeline = ResearchPipeline(model_type='xgboost', n_features=30)

        train_clean = train_df[selected_features + [target_cls]].dropna()
        test_clean = test_df[selected_features + [target_cls]].dropna()

        if len(train_clean) < 1000 or len(test_clean) < 100:
            continue

        pipeline.fit(train_clean[selected_features], train_clean[target_cls])

        y_pred = pipeline.predict(test_clean[selected_features])
        acc = accuracy_score(test_clean[target_cls], y_pred)

        up_ratio = test_clean[target_cls].mean()
        majority_acc = max(up_ratio, 1 - up_ratio)

        results.append({
            "Window": win_name,
            "Start": start.date(),
            "End": end.date(),
            "Accuracy": acc,
            "Majority_Acc": majority_acc,
            "Alpha": acc - majority_acc
        })

        logger.info(f"  Accuracy: {acc:.2%}, Majority: {majority_acc:.2%}, Alpha: {acc - majority_acc:.2%}")

    # Add final holdout for comparison
    fh_test = global_df[global_df.index >= final_holdout_start]
    fh_train = global_df[global_df.index < (final_holdout_start - timedelta(days=20))]
    fh_pipeline = ResearchPipeline(model_type='xgboost', n_features=30)
    fh_train_c = fh_train[selected_features + [target_cls]].dropna()
    fh_test_c = fh_test[selected_features + [target_cls]].dropna()
    fh_pipeline.fit(fh_train_c[selected_features], fh_train_c[target_cls])
    fh_acc = accuracy_score(fh_test_c[target_cls], fh_pipeline.predict(fh_test_c[selected_features]))
    fh_maj = max(fh_test_c[target_cls].mean(), 1 - fh_test_c[target_cls].mean())

    results.append({
        "Window": "PHASE 4 HOLDOUT",
        "Start": final_holdout_start.date(),
        "End": global_df.index.max().date(),
        "Accuracy": fh_acc,
        "Majority_Acc": fh_maj,
        "Alpha": fh_acc - fh_maj
    })

    pd.DataFrame(results).to_csv(DATA_REPORTS_DIR / "PHASE5_ROLLING_HOLDOUTS.csv", index=False)
    print("\nRolling Holdouts Summary:")
    print(pd.DataFrame(results))

if __name__ == "__main__":
    main()
