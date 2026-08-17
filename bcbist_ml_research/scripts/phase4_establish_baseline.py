import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import STOCK_UNIVERSE, DATA_FEATURES_DIR, DATA_REPORTS_DIR
from app.validation.walk_forward import WalkForwardValidation, purge_overlap
from app.ml.classification import DirectionalModel
from app.ml.baseline import BaselineModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BaselineEstablishment")

def clean_for_ml(df: pd.DataFrame, feature_cols: list):
    df = df.copy()
    X = df[feature_cols]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X

def main():
    logger.info("Establishing Frozen Baseline for Phase 4")

    # Load data once
    all_data_list = []
    for entry in STOCK_UNIVERSE:
        symbol = entry['symbol']
        path = DATA_FEATURES_DIR / f"{symbol.replace('.', '_')}_features.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, index_col='date', parse_dates=True)
            df['symbol'] = symbol
            df['sector'] = entry['sector']
            all_data_list.append(df)

    if not all_data_list:
        logger.error("No features found.")
        return

    global_df = pd.concat(all_data_list).sort_index()

    horizons = [1, 3, 5, 10, 20]
    results = []

    wf = WalkForwardValidation(n_folds=3, train_size_days=252, test_size_days=60, purge_days=20)

    drop_cols = ['open', 'high', 'low', 'close', 'volume', 'adj_close', 'symbol', 'sector']
    all_target_cols = [c for c in global_df.columns if c.startswith('target_')]
    feature_cols = [c for c in global_df.columns if c not in drop_cols and c not in all_target_cols]

    for h in horizons:
        target_col = f'target_up_{h}d'
        if target_col not in global_df.columns:
            logger.warning(f"Target {target_col} missing. Skipping.")
            continue

        logger.info(f"Training Baseline for Horizon: {h}d")
        splits = wf.split(global_df)

        fold_accs = []
        fold_f1s = []

        for fold, (train_df, test_df) in enumerate(splits):
            train_df = purge_overlap(train_df, test_df, h)
            train_clean = train_df.dropna(subset=[target_col])
            test_clean = test_df.dropna(subset=[target_col])

            if train_clean.empty or test_clean.empty:
                continue

            X_train = clean_for_ml(train_clean, feature_cols)
            y_train = train_clean[target_col]
            X_test = clean_for_ml(test_clean, feature_cols)
            y_test = test_clean[target_col]

            model = DirectionalModel(model_type="xgboost")
            model.train(X_train, y_train)

            y_pred = model.predict(X_test)
            fold_accs.append(accuracy_score(y_test, y_pred))
            fold_f1s.append(f1_score(y_test, y_pred))

        results.append({
            "Horizon": f"{h}d",
            "Accuracy": np.mean(fold_accs) if fold_accs else 0,
            "F1": np.mean(fold_f1s) if fold_f1s else 0,
            "Timestamp": datetime.now().isoformat()
        })

    baseline_df = pd.DataFrame(results)
    path = DATA_REPORTS_DIR / "PHASE4_BASELINE_RESULTS.csv"
    baseline_df.to_csv(path, index=False)
    logger.info(f"Baseline established and saved to {path}")

if __name__ == "__main__":
    main()
