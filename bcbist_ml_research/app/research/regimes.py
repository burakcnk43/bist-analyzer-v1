import pandas as pd
import numpy as np
import logging
import xgboost as xgb
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class MarketRegimeDetector:
    """
    Classifies market state based on breadth, volatility, and trend.
    Supports 10 granular states for Phase 16.
    """
    def __init__(self, rsi_window=14, vol_window=20):
        self.rsi_window = rsi_window
        self.vol_window = vol_window
        self.regimes = [
            "BULL", "BULL_OVEREXTENDED", "BULL_TO_SIDEWAYS",
            "SIDEWAYS_HIGH_VOL", "SIDEWAYS_LOW_VOL",
            "SIDEWAYS_TO_BEAR", "BEAR", "BEAR_TO_CRASH",
            "CRASH", "CRASH_TO_RECOVERY", "RECOVERY"
        ]

    def detect_regime(self, market_df: pd.DataFrame) -> str:
        """
        Expects market_df with indicators and BREADTH.
        """
        if market_df.empty: return "SIDEWAYS_LOW_VOL"

        row = market_df.iloc[-1]

        mkt_ret_1d = row.get('macro_bist100_return', 0)
        mkt_ret_5d = row.get('market_ret_5d', 0)
        if 'market_ret_5d' not in market_df.columns and 'macro_bist100_return' in market_df.columns:
            mkt_ret_5d = market_df['macro_bist100_return'].tail(5).sum()

        z_rsi = row.get('market_z_rsi_14', 0)
        z_vol = row.get('market_z_rolling_std_20', 0)

        # Breadth Divergence (Market UP, Breadth DOWN)
        breadth = row.get('pct_above_sma50', 0.5)

        # 1. Extreme States
        if z_rsi < -2.5 and mkt_ret_5d < -0.10:
            return "CRASH"
        if z_rsi > 2.5:
            return "BULL_OVEREXTENDED"

        # 2. Transitions
        if z_rsi < -2.0 and mkt_ret_1d > 0.02:
            return "CRASH_TO_RECOVERY"
        if mkt_ret_5d > 0.02 and mkt_ret_1d < -0.015:
            return "BULL_TO_SIDEWAYS"
        if mkt_ret_5d < -0.02 and mkt_ret_5d > -0.06 and z_vol > 1.2:
            return "SIDEWAYS_TO_BEAR"

        # 3. Stable States
        if mkt_ret_5d > 0.04 and breadth > 0.6:
            return "BULL"
        if mkt_ret_5d < -0.04 and breadth < 0.4:
            return "BEAR"
        if mkt_ret_5d > 0.01 and z_rsi < -1.0:
            return "RECOVERY"

        # 4. Sideways Differentiation
        if z_vol > 1.0:
            return "SIDEWAYS_HIGH_VOL"

        return "SIDEWAYS_LOW_VOL"

class RegimeTransitionModel:
    """
    Predicts specific state shift probabilities.
    """
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            objective='multi:softprob',
            num_class=9,
            random_state=42
        )
        self.features = [
            'market_z_rsi_14', 'market_z_rolling_std_20',
            'macro_bist100_return', 'macro_usdtry_return',
            'pct_above_sma50', 'advancing_declining_ratio'
        ]
        self.regime_map = {
            "BULL": 0, "BULL_TO_SIDEWAYS": 1, "SIDEWAYS": 2,
            "SIDEWAYS_TO_BEAR": 3, "BEAR": 4, "BEAR_TO_CRASH": 5,
            "CRASH": 6, "CRASH_TO_RECOVERY": 7, "RECOVERY": 8
        }
        self.rev_map = {v: k for k, v in self.regime_map.items()}

    def train(self, market_history: pd.DataFrame):
        """
        Target: What is the regime in 5 days?
        """
        df = market_history.copy()
        detector = MarketRegimeDetector()

        # Calculate daily regimes
        df['regime'] = [detector.detect_regime(df.iloc[:i+1]) for i in range(len(df))]
        df['target_regime'] = df['regime'].shift(-5)

        X = df[self.features].fillna(0)
        y = df['target_regime'].map(self.regime_map).fillna(2) # Default to Sideways

        logger.info(f"Training RegimeTransitionModel for 9 states on {len(X)} days...")
        self.model.fit(X, y)

    def predict_transition_matrix(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Returns probabilities for all 9 states in 5 days.
        """
        available = [f for f in self.features if f in X.columns]
        X_clean = X[available].tail(1).fillna(0)
        if X_clean.empty:
            return {r: 1.0/9.0 for r in self.regime_map}

        probs = self.model.predict_proba(X_clean)[0]
        return {self.rev_map[i]: float(probs[i]) for i in range(len(probs))}

    def predict_transition_prob(self, X: pd.DataFrame) -> float:
        """
        Backward compatibility: prob that current regime CHANGES.
        """
        detector = MarketRegimeDetector()
        current = detector.detect_regime(X)
        matrix = self.predict_transition_matrix(X)

        prob_stay = matrix.get(current, 0.0)
        return 1.0 - prob_stay
