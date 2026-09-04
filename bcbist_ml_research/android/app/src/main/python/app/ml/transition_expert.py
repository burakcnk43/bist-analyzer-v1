import pandas as pd
import numpy as np
import xgboost as xgb
import logging
import joblib
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class RegimeTransitionExpert:
    """
    Predicts the probability of market regime transitions.
    Uses macro indicators, global indices, and cross-sectional dispersion.
    """
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=42
        )
        self.feature_cols = []
        self.is_trained = False

    def prepare_transition_features(self, market_df: pd.DataFrame) -> pd.DataFrame:
        """
        Features:
        - Breadth momentum
        - Volatility acceleration
        - Cross-sector dispersion
        """
        df = pd.DataFrame(index=market_df.index)

        # 1. Breadth Dynamics
        if 'pct_above_sma50' in market_df.columns:
            df['breadth_mom'] = market_df['pct_above_sma50'].diff(5)
            df['breadth_level'] = market_df['pct_above_sma50']

        if 'breadth_momentum' in market_df.columns:
            df['b_mom_raw'] = market_df['breadth_momentum']

        # 2. Volatility Dynamics
        if 'market_z_rolling_std_20' in market_df.columns:
            df['vol_mom'] = market_df['market_z_rolling_std_20'].diff(5)
            df['vol_level'] = market_df['market_z_rolling_std_20']
        elif 'breadth_extreme' in market_df.columns:
            df['vol_proxy'] = market_df['breadth_extreme']

        # 3. Macro & Global
        if 'usdtry_return_5d' in market_df.columns:
            df['usdtry_ret'] = market_df['usdtry_return_5d']

        return df.fillna(0)

    def train(self, market_df: pd.DataFrame, regime_labels: pd.Series):
        """
        Target: 1 if regime(T+5) != regime(T), else 0.
        """
        X = self.prepare_transition_features(market_df)

        # Binary target: Shift in regime
        y = (regime_labels.shift(-5) != regime_labels).astype(int)

        # Valid indices (remove nan from shift)
        valid = X.index[:-5]
        self.feature_cols = X.columns.tolist()

        logger.info(f"Training RegimeTransitionExpert on {len(valid)} days...")
        self.model.fit(X.loc[valid], y.loc[valid])
        self.is_trained = True

    def predict_transition_prob(self, market_row: pd.DataFrame) -> float:
        """
        Returns probability of a transition occurring in the next 5 days.
        """
        if not self.is_trained:
            return 0.1 # Default low

        X = self.prepare_transition_features(market_row)
        return float(self.model.predict_proba(X)[:, 1][0])

    def save(self, path: Path):
        joblib.dump({
            'model': self.model,
            'feature_cols': self.feature_cols,
            'is_trained': self.is_trained
        }, path)
        logger.info(f"RegimeTransitionExpert saved to {path}")

    @classmethod
    def load(cls, path: Path):
        data = joblib.load(path)
        obj = cls()
        obj.model = data['model']
        obj.feature_cols = data['feature_cols']
        obj.is_trained = data['is_trained']
        return obj
