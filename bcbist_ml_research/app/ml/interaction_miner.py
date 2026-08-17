import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Dict
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

class FeatureInteractionEngine:
    """
    Discovers non-linear interactions between stable features.
    Filters for pairs that have consistent Information Coefficient (IC) across regimes.
    """
    def __init__(self, min_ic=0.02, stability_threshold=0.5):
        self.min_ic = min_ic
        self.stability_threshold = stability_threshold
        self.top_interactions = []

    def discover_interactions(self,
                               df: pd.DataFrame,
                               target_col: str,
                               base_features: List[str],
                               regime_col: str) -> List[Tuple[str, str]]:
        """
        Tests interaction pairs (Product or Ratio) and measures IC stability.
        """
        logger.info(f"Mining interactions between {len(base_features)} base features...")

        candidates = []
        regimes = df[regime_col].unique()

        for i, f1 in enumerate(base_features):
            for f2 in base_features[i+1:]:
                # Test Product
                inter_name = f"{f1}_x_{f2}"
                inter_val = df[f1] * df[f2]

                # Measure IC stability across regimes
                regime_ics = []
                for r in regimes:
                    mask = df[regime_col] == r
                    if mask.sum() < 100: continue
                    ic, _ = spearmanr(inter_val[mask], df[target_col][mask])
                    regime_ics.append(ic)

                if regime_ics:
                    mean_ic = np.mean(regime_ics)
                    std_ic = np.std(regime_ics) + 1e-9
                    stability = abs(mean_ic) / std_ic

                    if abs(mean_ic) >= self.min_ic and stability >= self.stability_threshold:
                        candidates.append({
                            'pair': (f1, f2),
                            'name': inter_name,
                            'mean_ic': mean_ic,
                            'stability': stability
                        })

        # Sort and return top interactions
        res_df = pd.DataFrame(candidates).sort_values('stability', ascending=False).head(10)
        self.top_interactions = res_df['pair'].tolist()

        logger.info(f"Discovered {len(self.top_interactions)} stable interactions.")
        return self.top_interactions

    def apply_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds discovered interaction columns to a DataFrame.
        """
        df_copy = df.copy()
        for f1, f2 in self.top_interactions:
            df_copy[f"{f1}_x_{f2}"] = df_copy[f1] * df_copy[f2]
        return df_copy
