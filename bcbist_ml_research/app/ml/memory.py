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
        else:
            self.prediction_log = pd.DataFrame(columns=[
                'date', 'symbol', 'sector', 'raw_score', 'market_regime',
                'market_quality', 'actual_return_1d', 'outcome_class'
            ])

    def save_memory(self):
        self.prediction_log.to_csv(self.storage_path, index=False)
        logger.info(f"Memory saved to {self.storage_path}")

    def add_predictions(self, df: pd.DataFrame):
        """
        Stores predictions with full context for Phase 15.
        """
        new_data = df.copy()
        if 'outcome_class' not in new_data.columns:
            new_data['outcome_class'] = 'PENDING'
        if 'actual_return_1d' not in new_data.columns:
            new_data['actual_return_1d'] = np.nan

        if not self.prediction_log.empty:
            # Ensure columns exist in log
            for col in new_data.columns:
                if col not in self.prediction_log.columns:
                    self.prediction_log[col] = np.nan

            combined = pd.concat([self.prediction_log, new_data])
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

    def update_outcomes(self, outcomes_df: pd.DataFrame, horizon='1d'):
        """
        Expects outcomes_df with [date, symbol, actual_return_1d]
        Optimized to prevent memory explosion.
        """
        if self.prediction_log.empty or outcomes_df.empty: return

        # Force string alignment for safe merging
        log = self.prediction_log.copy()
        log['date'] = pd.to_datetime(log['date'], format='ISO8601', errors='coerce').dt.strftime('%Y-%m-%d')
        log = log.drop_duplicates(subset=['date', 'symbol'])

        outcomes = outcomes_df.copy()
        outcomes['date'] = pd.to_datetime(outcomes['date'], format='ISO8601', errors='coerce').dt.strftime('%Y-%m-%d')
        outcomes = outcomes.drop_duplicates(subset=['date', 'symbol'])

        return_col = f'actual_return_{horizon}'

        # Merge outcomes safely
        merged = log.merge(
            outcomes[['date', 'symbol', return_col]],
            on=['date', 'symbol'],
            how='left',
            suffixes=('', '_new')
        )

        if f'{return_col}_new' in merged.columns:
            merged[return_col] = merged[return_col].combine_first(merged[f'{return_col}_new'])
            merged = merged.drop(columns=[f'{return_col}_new'])

        self.prediction_log = merged

        # Classify Outcome (Precision Sniper 1D Target)
        def classify(ret):
            if pd.isna(ret) or ret == 0: return 'PENDING'
            if ret > 0.01: return 'SUCCESS'
            if ret > 0.0: return 'PARTIAL_SUCCESS'
            return 'FAILURE'

        return_col = f'actual_return_{horizon}'
        if return_col in self.prediction_log.columns:
            self.prediction_log['outcome_class'] = self.prediction_log[return_col].apply(classify)

    def get_recent_performance(self, days=20) -> Dict:
        if self.prediction_log.empty: return {}

        log = self.prediction_log.copy()
        log['date'] = pd.to_datetime(log['date'], format='ISO8601', errors='coerce')

        # Filter out invalid dates
        log = log.dropna(subset=['date'])
        if log.empty: return {}

        recent = log[log['date'] >= (log['date'].max() - pd.Timedelta(days=days))]
        if recent.empty: return {}

        valid_outcomes = recent[recent['outcome_class'] != 'PENDING']
        if valid_outcomes.empty:
             return {"hit_rate": 0.5, "avg_return": 0.0, "success_ratio": 0.0}

        return {
            "hit_rate": (valid_outcomes['outcome_class'].isin(['SUCCESS', 'PARTIAL_SUCCESS'])).mean(),
            "avg_return": valid_outcomes.get('actual_return_1d', pd.Series([0])).mean(),
            "success_ratio": (valid_outcomes['outcome_class'] == 'SUCCESS').mean()
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
