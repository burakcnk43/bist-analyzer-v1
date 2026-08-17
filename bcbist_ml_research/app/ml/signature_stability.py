import pandas as pd
import numpy as np
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class StabilityEngine:
    """
    Ranks features and signals by their Regime-Consistency.
    Stable signals must perform well across both BULL and SIDEWAYS regimes.
    """
    def __init__(self, regimes=['BULL', 'SIDEWAYS_LOW_VOL', 'RECOVERY']):
        self.regimes = regimes

    def calculate_stability_scores(self,
                                   feature_X: pd.DataFrame,
                                   y: pd.Series,
                                   regime_labels: pd.Series) -> pd.DataFrame:
        """
        Returns a DataFrame of features with consistency scores.
        """
        results = []

        for col in feature_X.columns:
            if not pd.api.types.is_numeric_dtype(feature_X[col]): continue

            regime_corrs = {}
            for regime in self.regimes:
                mask = regime_labels == regime
                if mask.sum() < 50: continue

                corr = feature_X[col][mask].corr(y[mask])
                regime_corrs[regime] = corr

            if not regime_corrs: continue

            # Stability = mean correlation / std(correlations)
            corrs = list(regime_corrs.values())
            mean_c = np.mean(corrs)
            std_c = np.std(corrs) + 1e-9

            results.append({
                'feature': col,
                'mean_corr': mean_c,
                'stability_score': abs(mean_c) / std_c,
                'regime_count': len(corrs)
            })

        return pd.DataFrame(results).sort_values('stability_score', ascending=False)
