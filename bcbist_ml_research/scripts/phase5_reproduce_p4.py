import sys
import os
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import STOCK_UNIVERSE, DATA_FEATURES_DIR, DATA_REPORTS_DIR
from app.ml.pipeline import ResearchPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Phase5Reproduction")

def main():
    logger.info("--- PHASE 5: REPRODUCING PHASE 4 20D RESULT ---")

    # 1. Load All Features
    all_data_list = []
    for entry in STOCK_UNIVERSE:
        symbol = entry['symbol']
        path = DATA_FEATURES_DIR / f"{symbol.replace('.', '_')}_features.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, index_col='date', parse_dates=True)
            df['symbol_col'] = symbol
            df['sector_col'] = entry['sector']
            all_data_list.append(df)

    global_df = pd.concat(all_data_list).sort_index()

    # 2. Define Strict Holdout (Last 6 Months)
    last_date = global_df.index.max()
    holdout_start = last_date - timedelta(days=180)
    research_df = global_df[global_df.index < holdout_start]
    holdout_df = global_df[global_df.index >= holdout_start]

    # 3. Configure 20D FULL MODEL
    h = 20
    target_cls = f'target_up_{h}d'
    groups = {
        "Technical": ['rsi_', 'sma_', 'macd', 'bb_', 'dist_', 'atr_'],
        "Volume": ['volume', 'obv'],
        "Sector": ['rel_sector', 'sector_mean'],
        "Macro": ['macro_'],
        "KAP": ['event_', 'days_since']
    }
    drop_cols = ['open', 'high', 'low', 'close', 'volume', 'adj_close', 'symbol_col', 'sector_col']
    all_target_cols = [c for c in global_df.columns if c.startswith('target_')]

    selected_features = []
    for g in groups.keys():
        selected_features.extend([c for c in global_df.columns if any(p in c for p in groups[g])
                                 and c not in drop_cols and c not in all_target_cols])
    selected_features = sorted(list(set(selected_features)))

    logger.info(f"Training on Research Data: {research_df.index.min().date()} to {research_df.index.max().date()}")
    logger.info(f"Evaluating on Holdout: {holdout_df.index.min().date()} to {holdout_df.index.max().date()}")

    # 4. Train and Predict
    pipeline = ResearchPipeline(model_type='xgboost', n_features=30)

    train_data = research_df[selected_features + [target_cls]].dropna()
    pipeline.fit(train_data[selected_features], train_data[target_cls])

    test_data = holdout_df[selected_features + [target_cls]].dropna()
    y_true = test_data[target_cls]
    y_pred = pipeline.predict(test_data[selected_features])

    acc = accuracy_score(y_true, y_pred)

    logger.info(f"REPRODUCED ACCURACY: {acc:.4%}")

    # 5. Class Balance Check
    up_ratio = y_true.mean()
    majority_acc = max(up_ratio, 1 - up_ratio)

    print("\n" + "="*40)
    print("PHASE 4 REPRODUCTION AUDIT")
    print("="*40)
    print(f"Accuracy:          {acc:.2%}")
    print(f"Majority Accuracy:  {majority_acc:.2%}")
    print(f"Observations:      {len(y_true)}")
    print(f"UP Class Count:    {int(y_true.sum())}")
    print(f"DOWN Class Count:  {len(y_true) - int(y_true.sum())}")
    print("-"*40)
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("-"*40)
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print("="*40)

    # Save Frozen Config
    config = {
        "horizon": h,
        "features_count": len(selected_features),
        "selected_features": selected_features,
        "n_top_features": 30,
        "holdout_start": holdout_start.isoformat(),
        "reproduced_accuracy": acc,
        "timestamp": datetime.now().isoformat()
    }
    with open(DATA_REPORTS_DIR / "PHASE5_FROZEN_PHASE4_CONFIG.json", "w") as f:
        json.dump(config, f, indent=4)

if __name__ == "__main__":
    main()
