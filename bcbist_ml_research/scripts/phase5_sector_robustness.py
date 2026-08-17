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
logger = logging.getLogger("SectorRobustness")

def main():
    logger.info("--- PHASE 5: SECTOR ROBUSTNESS AUDIT ---")

    # 1. Load Data
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

    holdout_start = global_df.index.max() - timedelta(days=180)
    research_df = global_df[global_df.index < holdout_start]
    holdout_df = global_df[global_df.index >= holdout_start]

    h = 20
    target_cls = f'target_up_{h}d'

    feature_patterns = ['rsi_', 'sma_', 'macd', 'bb_', 'dist_', 'atr_', 'volume', 'obv', 'rel_sector', 'sector_mean', 'macro_', 'event_']
    drop_cols = ['open', 'high', 'low', 'close', 'volume', 'adj_close', 'symbol_col', 'sector_col']
    selected_features = sorted(list(set([c for c in global_df.columns if any(p in c for p in feature_patterns)
                                       and c not in drop_cols and not c.startswith('target_')])))

    # Train Global Model
    logger.info("Training Global Model...")
    pipeline = ResearchPipeline(model_type='xgboost', n_features=30)
    train_clean = research_df[selected_features + [target_cls]].dropna()
    pipeline.fit(train_clean[selected_features], train_clean[target_cls])

    # Sector Evaluation
    sectors = global_df['sector_col'].unique()
    results = []

    for sector in sectors:
        sector_test = holdout_df[holdout_df['sector_col'] == sector]
        sector_test_clean = sector_test[selected_features + [target_cls]].dropna()

        if len(sector_test_clean) < 20: continue

        y_pred = pipeline.predict(sector_test_clean[selected_features])
        acc = accuracy_score(sector_test_clean[target_cls], y_pred)

        up_ratio = sector_test_clean[target_cls].mean()
        majority_acc = max(up_ratio, 1 - up_ratio)

        results.append({
            "Sector": sector,
            "N": len(sector_test_clean),
            "Accuracy": acc,
            "Majority_Acc": majority_acc,
            "Alpha": acc - majority_acc
        })

    res_df = pd.DataFrame(results).sort_values("Alpha", ascending=False)
    res_df.to_csv(DATA_REPORTS_DIR / "PHASE5_SECTOR_ROBUSTNESS.csv", index=False)

    print("\nSector Robustness Summary:")
    print(res_df.to_markdown(index=False))

    # Also check Stock Level for top sector
    logger.info("Top Sector breakdown...")
    top_sector = res_df.iloc[0]['Sector']
    # ... logic for stock level ...

if __name__ == "__main__":
    main()
