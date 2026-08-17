import pandas as pd
import numpy as np
from datetime import timedelta
from typing import List, Tuple

class WalkForwardValidation:
    def __init__(self, n_folds=5, train_size_days=365, test_size_days=60, purge_days=20):
        self.n_folds = n_folds
        self.train_size_days = train_size_days
        self.test_size_days = test_size_days
        self.purge_days = purge_days

    def split(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generates (train, test) splits.
        """
        splits = []
        df = df.sort_index()
        all_dates = df.index.unique()

        if len(all_dates) < self.train_size_days + self.test_size_days:
            # Fallback if dataset is small
            mid = int(len(df) * 0.8)
            return [(df.iloc[:mid - self.purge_days], df.iloc[mid:])]

        # Calculate splits from the end
        last_date = all_dates[-1]

        for i in range(self.n_folds):
            test_end = last_date - timedelta(days=i * self.test_size_days)
            test_start = test_end - timedelta(days=self.test_size_days)

            train_end = test_start - timedelta(days=self.purge_days)
            train_start = train_end - timedelta(days=self.train_size_days)

            train_df = df.loc[(df.index >= train_start) & (df.index <= train_end)]
            test_df = df.loc[(df.index >= test_start) & (df.index <= test_end)]

            if not train_df.empty and not test_df.empty:
                splits.append((train_df, test_df))

        return splits[::-1] # Chronological order

def purge_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """
    Ensures no training row has a target that overlaps with the test period.
    """
    test_start = test_df.index.min()
    # Any training row whose target_date (current_date + horizon) >= test_start is purged
    # Effectively, train_end must be < test_start - horizon
    limit = test_start - timedelta(days=horizon_days)
    return train_df.loc[train_df.index < limit]
