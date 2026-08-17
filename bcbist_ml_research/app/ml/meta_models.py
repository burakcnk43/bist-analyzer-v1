import pandas as pd
import numpy as np
import xgboost as xgb
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class SuccessPredictor:
    def __init__(self):
        self.model = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
        self.features = []

    def train(self, log_df: pd.DataFrame, available_features: List[str]):
        train_data = log_df[log_df['daily_rank'] <= 20].copy()
        numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
        exclude = ['target', 'hit', 'symbol', 'sector', 'rank', 'col', 'is_top5', 'date', 'Unnamed']
        self.features = [f for f in available_features if f in numeric_cols and not any(x in f for x in exclude)]

        # Add prediction context
        for cf in ['alpha_score', 'daily_rank']:
            if cf in train_data.columns: self.features.append(cf)

        X = train_data[self.features].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = (train_data['target_return_5d'] > 0).astype(int)
        self.model.fit(X, y)

    def predict_trust(self, X: pd.DataFrame) -> np.ndarray:
        X_clean = X[self.features].replace([np.inf, -np.inf], np.nan).fillna(0)
        return self.model.predict_proba(X_clean)[:, 1]

class FailurePredictor:
    def __init__(self):
        self.model = xgb.XGBClassifier(n_estimators=30, max_depth=2, random_state=42)
        self.features = ['macro_usdtry_return', 'macro_bist100_return', 'market_z_rsi_14']

    def train(self, daily_log_df: pd.DataFrame):
        y = (daily_log_df['hit_rate'] <= 0.4).astype(int)
        X = daily_log_df[self.features].replace([np.inf, -np.inf], np.nan).fillna(0)

        logger.info(f"Training FailurePredictor on {len(X)} days...")
        self.model.fit(X, y)

    def predict_failure_prob(self, X: pd.DataFrame) -> np.ndarray:
        X_clean = X[self.features].replace([np.inf, -np.inf], np.nan).fillna(0)
        return self.model.predict_proba(X_clean)[:, 1]
