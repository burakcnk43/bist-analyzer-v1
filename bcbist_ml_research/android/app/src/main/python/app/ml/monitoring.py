import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class DriftGuard:
    """
    Monitors Population Stability Index (PSI) to detect feature drift.
    """
    def __init__(self, threshold=0.2):
        self.threshold = threshold
        self.reference_stats = {} # feature -> bins

    def fit_reference(self, train_df: pd.DataFrame, features: List[str]):
        """
        Learns training distribution for PSI calculation.
        """
        for f in features:
            if f not in train_df.columns: continue
            # Create 10 deciles as reference
            data = train_df[f].dropna()
            if len(data) > 10:
                self.reference_stats[f] = np.percentile(data, np.linspace(0, 100, 11))

    def calculate_psi(self, current_df: pd.DataFrame) -> float:
        """
        Calculates average PSI across all tracked features.
        """
        psis = []
        for f, bins in self.reference_stats.items():
            if f not in current_df.columns: continue

            # Reference distribution (uniform 10% per decile)
            expected = np.full(10, 0.1)

            # Actual distribution in current pool
            actual_counts, _ = np.histogram(current_df[f].dropna(), bins=bins)
            actual = actual_counts / (np.sum(actual_counts) + 1e-9)
            # Clip for log
            actual = np.clip(actual, 0.001, 1.0)

            psi = np.sum((actual - expected) * np.log(actual / expected))
            psis.append(psi)

        return float(np.mean(psis)) if psis else 0.0

class DriftGuardV2(DriftGuard):
    """
    Advanced monitoring that tracks Feature, Target, and Relationship drift.
    """
    def __init__(self, threshold=0.2):
        super().__init__(threshold)
        self.reference_corrs = {} # feature -> corr with target

    def fit_relationship_reference(self, df: pd.DataFrame, target_col: str, features: List[str]):
        """
        Learns baseline correlations between features and targets.
        """
        for f in features:
            if f in df.columns:
                self.reference_corrs[f] = df[f].corr(df[target_col])

    def calculate_relationship_drift(self, current_df: pd.DataFrame, current_target: pd.Series) -> float:
        """
        Measures if feature-target relationships have inverted or collapsed.
        """
        drifts = []
        for f, ref_corr in self.reference_corrs.items():
            if f in current_df.columns:
                cur_corr = current_df[f].corr(current_target)
                if not pd.isna(cur_corr):
                    # Drift = Abs difference in correlation
                    drifts.append(abs(cur_corr - ref_corr))

        return float(np.mean(drifts)) if drifts else 0.0

    def get_global_drift_score(self, current_df: pd.DataFrame, current_target: Optional[pd.Series] = None) -> float:
        """
        Combines PSI (Feature Drift) and Relationship Drift.
        """
        psi = self.calculate_psi(current_df)
        rel_drift = 0.0
        if current_target is not None:
            rel_drift = self.calculate_relationship_drift(current_df, current_target)

        # Weighted average
        return 0.6 * psi + 0.4 * rel_drift

    def get_drift_penalty(self, current_pool: pd.DataFrame) -> float:
        """
        Returns a penalty multiplier [0-1].
        """
        score = self.get_global_drift_score(current_pool)
        if score > self.threshold:
            penalty = min(0.5, (score - self.threshold) * 2.0)
            return penalty
        return 0.0
