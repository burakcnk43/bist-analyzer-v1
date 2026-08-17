import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import STOCK_UNIVERSE, DATA_FEATURES_DIR
from app.validation.walk_forward import WalkForwardValidation, purge_overlap
from app.ml.classification import DirectionalModel
from app.ml.baseline import BaselineModel
from app.ml.model_registry import ModelRegistry

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ModelTraining")

def clean_for_ml(df: pd.DataFrame, feature_cols: list):
    """
    Handles NaNs and Inf values for ML models.
    """
    df = df.copy()
    X = df[feature_cols]
    # Replace inf with large values or 0
    X = X.replace([np.inf, -np.inf], np.nan)
    # Fill remaining NaNs with 0 (or median, but 0 is safer for sparse financial features)
    X = X.fillna(0)
    return X

def main():
    logger.info("Starting Purged Walk-Forward Training & Validation")

    registry = ModelRegistry()
    wf = WalkForwardValidation(n_folds=3, train_size_days=252, test_size_days=60, purge_days=20)

    # 1. Load All Features into a Global Dataframe
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
        logger.error("No features found. Run build_features.py first.")
        return

    global_df = pd.concat(all_data_list).sort_index()

    # Identify feature and target columns
    target_col = 'target_up_5d'
    drop_cols = ['open', 'high', 'low', 'close', 'volume', 'adj_close', 'symbol', 'sector']
    all_target_cols = [c for c in global_df.columns if c.startswith('target_')]
    feature_cols = [c for c in global_df.columns if c not in drop_cols and c not in all_target_cols]

    # 2. Walk-Forward Loop for Global Model
    logger.info(f"Executing Global Model Walk-Forward on {len(feature_cols)} features...")
    splits = wf.split(global_df)

    global_results = []
    baseline_results = []

    for fold, (train_df, test_df) in enumerate(splits):
        logger.info(f"Fold {fold+1}/{len(splits)} | Train: {train_df.index.min().date()} to {train_df.index.max().date()}")

        # Purge
        train_df = purge_overlap(train_df, test_df, 5)

        # CLEAN DATA
        train_clean = train_df.dropna(subset=[target_col])
        test_clean = test_df.dropna(subset=[target_col])

        if train_clean.empty or test_clean.empty:
            logger.warning("Empty clean data in fold. Skipping.")
            continue

        X_train = clean_for_ml(train_clean, feature_cols)
        y_train = train_clean[target_col]

        X_test = clean_for_ml(test_clean, feature_cols)
        y_test = test_clean[target_col]

        logger.info(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples")

        # ML Model
        model = DirectionalModel(model_type="xgboost")
        model.train(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Baseline Model
        baseline = BaselineModel()
        y_base = baseline.predict_direction(test_clean)
        acc_base = accuracy_score(y_test, y_base)

        global_results.append(acc)
        baseline_results.append(acc_base)

        logger.info(f"Fold {fold+1} Accuracy: ML={acc:.4f}, Baseline={acc_base:.4f}")

    # 3. Final Summary
    logger.info("--- MODEL PERFORMANCE SUMMARY ---")
    if global_results:
        avg_acc = np.mean(global_results)
        avg_base = np.mean(baseline_results)
        logger.info(f"Global Model Avg Accuracy: {avg_acc:.4f}")
        logger.info(f"Baseline Avg Accuracy: {avg_base:.4f}")
        logger.info(f"Outperformance: {avg_acc - avg_base:.4f}")

    # Save the final model
    logger.info("Saving production model...")
    prod_clean = global_df.dropna(subset=[target_col]).tail(10000)
    prod_X = clean_for_ml(prod_clean, feature_cols)
    prod_y = prod_clean[target_col]

    final_model = DirectionalModel(model_type="xgboost")
    final_model.train(prod_X, prod_y)

    metadata = {
        "features": feature_cols,
        "avg_accuracy": np.mean(global_results) if global_results else 0,
        "baseline_accuracy": np.mean(baseline_results) if baseline_results else 0,
        "train_date": datetime.now().isoformat()
    }
    registry.save_model(final_model, metadata, "global_directional_5d")

if __name__ == "__main__":
    main()
