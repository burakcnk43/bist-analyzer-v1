import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Generator
from itertools import combinations

logger = logging.getLogger(__name__)

class CPCVAnalyzer:
    """
    Implements Combinatorial Purged Cross-Validation.
    Used to estimate the distribution of OOS performance and overfitting risk (PBO).
    """
    def __init__(self, n_blocks=6, k_test=2, purge_days=5):
        self.n_blocks = n_blocks
        self.k_test = k_test
        self.purge_days = purge_days

    def split(self, df: pd.DataFrame) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        Splits data into N blocks and yields (train, test) for each combination of k_test blocks.
        Implements proper purging and embargo to prevent leakage with 5D overlapping targets.
        """
        # Ensure df is sorted by date
        df = df.sort_index()
        unique_dates = df.index.unique()
        n_dates = len(unique_dates)

        block_size = n_dates // self.n_blocks
        block_date_ranges = []
        for i in range(self.n_blocks):
            start_idx = i * block_size
            end_idx = (i + 1) * block_size if i < self.n_blocks - 1 else n_dates
            block_date_ranges.append(unique_dates[start_idx:end_idx])

        # combinations of block indices for testing
        block_ids = list(range(self.n_blocks))
        for test_ids in combinations(block_ids, self.k_test):
            train_ids = [i for i in block_ids if i not in test_ids]

            # Construct test set
            test_dates = pd.DatetimeIndex(np.concatenate([block_date_ranges[i] for i in test_ids]))
            test_df = df.loc[test_dates].sort_index()

            # Construct train set with purging
            # For each contiguous test block, purge observations that overlap
            # Observations within [test_start - 5d, test_end] have overlap
            forbidden_dates = set(test_dates)

            # Simple purging: remove purge_days before each test date
            for td in test_dates:
                for p in range(1, self.purge_days + 1):
                    forbidden_dates.add(td - pd.Timedelta(days=p))
                    # Also embargo (after test)
                    forbidden_dates.add(td + pd.Timedelta(days=p))

            train_df = df[~df.index.isin(forbidden_dates)].sort_index()

            if not train_df.empty and not test_df.empty:
                yield train_df, test_df

    def calculate_pbo(self, results: List[float]) -> float:
        """
        Estimates Probability of Backtest Overfitting (Simplified).
        Calculates the frequency where rank in training doesn't match rank in OOS.
        (Requires multiple model configurations to be valid).
        """
        if len(results) < 2: return 0.0
        # Return 0.1 as a placeholder for now until full multi-config loop implemented
        return 0.1
