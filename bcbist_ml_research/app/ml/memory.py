import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class MemoryEngine:
    """
    Stores and retrieves historical prediction performance.
    Used to identify regime-specific strengths and weaknesses.
    """
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.prediction_log = pd.DataFrame()
        self._load_memory()

    def _load_memory(self):
        if self.storage_path.exists():
            self.prediction_log = pd.read_csv(self.storage_path, parse_dates=['date'])
            logger.info(f"Loaded {len(self.prediction_log)} memory entries.")

    def save_memory(self):
        self.prediction_log.to_csv(self.storage_path, index=False)
        logger.info(f"Memory saved to {self.storage_path}")

    def add_predictions(self, df: pd.DataFrame):
        """
        Stores predictions with full context for Phase 15.
        Columns might include:
        [date, symbol, sector, rank, final_prob, box_score, sector_score, regime_score,
         market_regime, market_breadth, vol_ratio, etc.]
        """
        # Ensure we don't have duplicates
        new_data = df.copy()
        if not self.prediction_log.empty:
            # Check for existing date/symbol pairs
            combined = pd.concat([self.prediction_log, new_data])
            # Keep newest record if same date/symbol
            self.prediction_log = combined.drop_duplicates(subset=['date', 'symbol'], keep='last')
        else:
            self.prediction_log = new_data

    def get_failure_patterns(self, min_samples=10) -> pd.DataFrame:
        """
        Mines the log for recurring failure conditions.
        """
        if self.prediction_log.empty: return pd.DataFrame()
        failures = self.prediction_log[self.prediction_log['outcome_class'] == 'FAILURE']
        if len(failures) < min_samples: return pd.DataFrame()

        # Analyze failures by regime/sector/box
        analysis = failures.groupby(['market_regime', 'sector']).size().reset_index(name='fail_count')
        return analysis.sort_values('fail_count', ascending=False)

    def update_outcomes(self, outcomes_df: pd.DataFrame):
        """
        Expects outcomes_df with [date, symbol, actual_return_5d]
        """
        if self.prediction_log.empty: return

        if not outcomes_df.empty:
            # Ensure date types match
            outcomes_df['date'] = pd.to_datetime(outcomes_df['date'])

            # Merge outcomes
            merged = self.prediction_log.merge(
                outcomes_df[['date', 'symbol', 'actual_return_5d']],
                on=['date', 'symbol'],
                how='left',
                suffixes=('', '_new')
            )

            # Update if exists
            if 'actual_return_5d_new' in merged.columns:
                merged['actual_return_5d'] = merged['actual_return_5d'].combine_first(merged['actual_return_5d_new'])
                merged = merged.drop(columns=['actual_return_5d_new'])

            self.prediction_log = merged

        # Classify Outcome
        def classify(ret):
            if pd.isna(ret): return 'PENDING'
            if ret > 0.02: return 'SUCCESS' # Strong win
            if ret > 0.0: return 'PARTIAL_SUCCESS'
            return 'FAILURE'

        self.prediction_log['outcome_class'] = self.prediction_log['actual_return_5d'].apply(classify)

    def get_recent_performance(self, days=20) -> Dict:
        if self.prediction_log.empty: return {}
        recent = self.prediction_log[self.prediction_log['date'] >= (self.prediction_log['date'].max() - pd.Timedelta(days=days))]
        if recent.empty: return {}

        return {
            "hit_rate": (recent['actual_return_5d'] > 0).mean(),
            "avg_return": recent['actual_return_5d'].mean(),
            "success_ratio": (recent['outcome_class'] == 'SUCCESS').mean()
        }

    def get_sector_reliability(self) -> pd.DataFrame:
        if self.prediction_log.empty: return pd.DataFrame()
        valid = self.prediction_log[self.prediction_log['outcome_class'] != 'PENDING']
        if valid.empty: return pd.DataFrame()

        stats = valid.groupby('sector').agg({
            'actual_return_5d': ['mean', 'count'],
            'raw_score': 'mean'
        })
        stats.columns = ['avg_return', 'sample_count', 'avg_score']
        return stats
