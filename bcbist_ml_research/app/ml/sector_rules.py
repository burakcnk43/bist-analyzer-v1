import pandas as pd
import numpy as np
import xgboost as xgb
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class RelativeStrengthSpecialist(SectorConditionalSuccessModel):
    """
    Refined specialist that predicts outperformance relative to sector mean.
    Helps identify leadership within groups.
    """
    def train_excess_models(self, X: pd.DataFrame, excess_return: pd.Series):
        """
        excess_return: stock_return - sector_mean_return
        """
        # Binary target: Did it beat its sector?
        y_excess = (excess_return > 0).astype(int)
        self.train_sector_models(X, y_excess)
    def __init__(self, min_samples=200):
        self.min_samples = min_samples
        self.sector_models = {} # sector -> model
        self.general_model = xgb.XGBClassifier(n_estimators=50, max_depth=3)
        self.feature_cols = []

    def train_sector_models(self, X: pd.DataFrame, y: pd.Series):
        """
        Trains individual models for each sector with enough data.
        """
        # Automatically detect and exclude non-numeric/string columns
        exclude = ['date', 'symbol', 'sector', 'target', 'symbol_col', 'regime']

        # Use pandas numeric type detection
        self.feature_cols = [c for c in X.columns if c not in exclude and pd.api.types.is_numeric_dtype(X[c])]

        if not self.feature_cols:
            logger.error("No numeric feature columns found for SectorConditionalSuccessModel!")
            return

        # 1. Train General Fallback
        logger.info(f"Training Sector Fallback model with {len(self.feature_cols)} features...")
        self.general_model.fit(X[self.feature_cols].fillna(0), y)

        # 2. Train Sector Specifics
        sectors = X['sector'].unique()
        for sector in sectors:
            mask = X['sector'] == sector
            X_sec = X[mask]
            y_sec = y[mask]

            if len(X_sec) >= self.min_samples:
                logger.info(f"Training Specialist for sector: {sector} ({len(X_sec)} samples)")
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.05,
                    random_state=42
                )
                model.fit(X_sec[self.feature_cols].fillna(0), y_sec)
                self.sector_models[sector] = model
            else:
                logger.debug(f"Skipping specialist for {sector}, insufficient data.")

    def predict_probs(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts using sector model if available, else general model.
        """
        probs = np.zeros(len(X))
        X_clean = X[self.feature_cols].fillna(0)

        for sector, model in self.sector_models.items():
            mask = X['sector'] == sector
            if mask.any():
                probs[mask] = model.predict_proba(X_clean[mask])[:, 1]

        # Fill remaining with general model
        remaining_mask = ~X['sector'].isin(self.sector_models.keys())
        if remaining_mask.any():
            probs[remaining_mask] = self.general_model.predict_proba(X_clean[remaining_mask])[:, 1]

        return probs
