import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import STOCK_UNIVERSE, DATA_FEATURES_DIR, DATA_REPORTS_DIR
from app.ml.pipeline import ResearchPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CalibrationAudit")

def main():
    logger.info("--- PHASE 5: PROBABILITY CALIBRATION AUDIT ---")

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

    # Train Global Model
    pipeline = ResearchPipeline(model_type='xgboost', n_features=30)
    train_clean = research_df[selected_features + [target_cls]].dropna()
    pipeline.fit(train_clean[selected_features], train_clean[target_cls])

    # Evaluation
    test_clean = holdout_df[selected_features + [target_cls]].dropna()
    y_true = test_clean[target_cls]
    # predict_proba ALREADY returns the class 1 probability
    y_prob = pipeline.predict_proba(test_clean[selected_features])

    # Brier Score
    brier = brier_score_loss(y_true, y_prob)

    # Confidence Buckets
    buckets = [0.0, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 1.0]
    bucket_results = []

    for i in range(len(buckets)-1):
        low = buckets[i]
        high = buckets[i+1]

        mask = (y_prob >= low) & (y_prob < high)
        subset_y = y_true[mask]

        if len(subset_y) > 0:
            bucket_results.append({
                "Range": f"{low:.2f}-{high:.2f}",
                "N": len(subset_y),
                "Avg_Prob": y_prob[mask].mean(),
                "Actual_Freq": subset_y.mean(),
                "Diff": y_prob[mask].mean() - subset_y.mean()
            })

    res_df = pd.DataFrame(bucket_results)
    res_df.to_csv(DATA_REPORTS_DIR / "PHASE5_CALIBRATION.csv", index=False)

    print("\nCalibration Audit Summary:")
    print(f"Brier Score: {brier:.4f}")
    print(res_df.to_markdown(index=False))

if __name__ == "__main__":
    main()
