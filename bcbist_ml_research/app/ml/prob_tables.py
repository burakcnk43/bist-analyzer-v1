import pandas as pd
import numpy as np
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class ConditionalProbEngine:
    """
    Calculates interpretable conditional probability tables.
    P(Success | FactorRange).
    Used to provide soft overrides to model predictions.
    """
    def __init__(self, min_samples=50):
        self.min_samples = min_samples
        self.tables = {} # factor -> {bin -> success_rate}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Learns success rates for various technical and market bins.
        """
        factors = ['rsi_14', 'market_z_rsi_14', 'relative_volume', 'pct_above_sma50', 'dist_sma_20']
        factors = [f for f in factors if f in X.columns]

        for factor in factors:
            # Create 10 bins for each factor
            bins = pd.qcut(X[factor], q=10, duplicates='drop')
            stats = y.groupby(bins, observed=False).agg(['mean', 'count'])

            # Only keep reliable bins
            valid_bins = stats[stats['count'] >= self.min_samples]
            if not valid_bins.empty:
                self.tables[factor] = valid_bins['mean'].to_dict()
                logger.info(f"Learned conditional prob table for {factor} ({len(valid_bins)} bins)")

    def get_adjustment_factor(self, row: pd.Series) -> float:
        """
        Returns a probability multiplier based on factor bins.
        Neutral = 1.0.
        """
        adjustments = []
        for factor, bin_probs in self.tables.items():
            val = row.get(factor)
            if pd.isna(val): continue

            # Find which bin it belongs to
            for bin_interval, prob in bin_probs.items():
                if val in bin_interval:
                    # Map probability to adjustment: 0.5 is neutral.
                    # Adjustment = prob / 0.5 (Base rate approximation)
                    adj = prob / 0.5
                    adjustments.append(adj)
                    break

        if not adjustments:
            return 1.0

        # Geomean of adjustments
        return float(np.power(np.prod(adjustments), 1.0/len(adjustments)))

    def apply_overrides(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """
        Adjusts production_alpha based on empirical bin probabilities.
        """
        df = candidates_df.copy()
        adjs = df.apply(self.get_adjustment_factor, axis=1)
        df['production_alpha'] = (df['production_alpha'] * adjs).clip(0, 1)
        return df
