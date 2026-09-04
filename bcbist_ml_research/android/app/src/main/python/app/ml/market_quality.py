import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class HighQualityMarketDayPredictor:
    """
    Estimates the 'Friendliness' of the current market environment
    for aggressive bullish selection.

    Factors: Breadth expansion, low volatility regimes, sector rotation stability.
    """
    def __init__(self):
        self.weights = {
            'breadth_mom': 0.3,
            'breadth_level': 0.3,
            'vol_stability': 0.2,
            'regime_quality': 0.2
        }

    def calculate_quality_score(self, market_row: pd.Series, regime: str) -> float:
        """
        Returns a score [0-100].
        """
        # 1. Breadth Level (0-1)
        b_level = market_row.get('pct_above_sma50', 0.5)

        # 2. Breadth Momentum (0-1)
        # Normalize: -0.1 to 0.1 -> 0 to 1
        b_mom = np.clip((market_row.get('breadth_momentum', 0) + 0.1) / 0.2, 0, 1)

        # 3. Volatility Stability (0-1)
        # Normalize: Z-vol 0 (stable) to 3 (panic)
        v_stab = np.clip(1.0 - (abs(market_row.get('market_z_rolling_std_20', 0)) / 3.0), 0, 1)

        # 4. Regime Quality (0-1)
        regime_scores = {
            'BULL': 1.0,
            'BULL_OVEREXTENDED': 0.6,
            'RECOVERY': 0.8,
            'SIDEWAYS_LOW_VOL': 0.7,
            'SIDEWAYS_HIGH_VOL': 0.4,
            'BEAR': 0.2,
            'CRASH': 0.0
        }
        r_qual = regime_scores.get(regime, 0.5)

        raw_score = (
            self.weights['breadth_level'] * b_level +
            self.weights['breadth_mom'] * b_mom +
            self.weights['vol_stability'] * v_stab +
            self.weights['regime_quality'] * r_qual
        )

        return float(np.clip(raw_score * 100, 0, 100))
