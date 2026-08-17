import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from app.ml.pipeline import ResearchPipeline
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

class SectorExpertManager:
    """
    Manages specialized models for individual BIST sectors.
    Fallback to general model if sector data is sparse.
    """
    def __init__(self, models_dir: Path, min_samples=1000):
        self.models_dir = models_dir
        self.min_samples = min_samples
        self.experts = {} # sector -> ResearchPipeline

    def train_experts(self, df: pd.DataFrame, features: List[str], target_col: str, task='ranking'):
        sectors = df['sector'].unique()

        for sector in sectors:
            sector_df = df[df['sector'] == sector].dropna(subset=[target_col])
            if len(sector_df) < self.min_samples:
                logger.info(f"Sector {sector} has insufficient samples ({len(sector_df)}). Skipping specialist.")
                continue

            logger.info(f"Training Sector Specialist: {sector} ({len(sector_df)} samples)")
            expert = ResearchPipeline(model_type='xgboost', n_features=30, task=task)
            expert.fit(sector_df[features], sector_df[target_col])

            self.experts[sector] = expert
            # Save
            safe_name = sector.lower().replace(" ", "_").replace("/", "_")
            joblib.dump(expert, self.models_dir / f"expert_sector_{safe_name}.joblib")

    def predict(self, X: pd.DataFrame, general_expert: ResearchPipeline) -> np.ndarray:
        """
        X must contain 'sector' column. Handles non-unique indices.
        """
        results = np.zeros(len(X))

        for sector, group in X.groupby('sector'):
            mask = (X['sector'] == sector).values

            if sector in self.experts:
                results[mask] = self.experts[sector].predict(group)
            else:
                results[mask] = general_expert.predict(group)

        return results

class RegimeExpertManager:
    """
    Manages specialized models for market regimes.
    """
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.experts = {} # regime -> ResearchPipeline

    def train_experts(self, df: pd.DataFrame, features: List[str], target_col: str, task='ranking'):
        # Expects 'regime' column
        if 'regime' not in df.columns:
            logger.error("Regime column missing from training data.")
            return

        regimes = df['regime'].unique()
        for reg in regimes:
            reg_df = df[df['regime'] == reg].dropna(subset=[target_col])
            if len(reg_df) < 500: continue

            logger.info(f"Training Regime Specialist: {reg} ({len(reg_df)} samples)")
            expert = ResearchPipeline(model_type='xgboost', n_features=30, task=task)
            expert.fit(reg_df[features], reg_df[target_col])
            self.experts[reg] = expert
            joblib.dump(expert, self.models_dir / f"expert_regime_{reg.lower()}.joblib")

    def predict(self, X: pd.DataFrame, regime: str, general_expert: ResearchPipeline) -> np.ndarray:
        if regime in self.experts:
            return self.experts[regime].predict(X)
        return general_expert.predict(X)
