import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Callable

logger = logging.getLogger(__name__)

class ResearchEvaluator:
    """
    Standardizes backtesting and evaluation of BCBIST research models.
    Supports Conditional vs Unconditional metrics and Adaptive K analysis.
    """
    def __init__(self, target_horizon=5):
        self.target_horizon = target_horizon

    def evaluate_strategy(self,
                          test_df: pd.DataFrame,
                          selection_func: Callable[[pd.Timestamp, pd.DataFrame], pd.DataFrame],
                          actuals_df: pd.DataFrame,
                          regime_detector=None,
                          breadth_df=None) -> Dict:
        """
        selection_func: (date, day_data) -> selected_stocks_df
        """
        dates = sorted(test_df.index.unique())
        results = []

        logger.info(f"Running standardized evaluation over {len(dates)} days...")

        for date in dates:
            day_data = test_df.loc[[date]]
            picks = selection_func(date, day_data)

            k = len(picks)
            hits = 0
            avg_ret = 0

            if k > 0:
                day_actuals = actuals_df.loc[[date]].set_index('symbol_col')
                picked_syms = picks['symbol_col'].tolist()

                valid_syms = [s for s in picked_syms if s in day_actuals.index]
                if valid_syms:
                    rets = day_actuals.loc[valid_syms]['target_return_5d']
                    hits = (rets > 0).sum()
                    avg_ret = rets.mean()

            regime = "UNKNOWN"
            if regime_detector and breadth_df is not None:
                day_mkt = breadth_df.loc[[date]]
                regime = regime_detector.detect_regime(day_mkt)

            results.append({
                'date': date,
                'k': k,
                'hits': hits,
                'is_3plus': int(hits >= 3),
                'is_1plus': int(hits >= 1),
                'avg_return': avg_ret,
                'regime': regime
            })

        res_df = pd.DataFrame(results)

        # Calculate Global Metrics
        total_days = len(res_df)
        active_days = (res_df['k'] > 0).sum()

        metrics = {
            'coverage': active_days / total_days if total_days > 0 else 0,
            'avg_k': res_df['k'].mean(),
            'unconditional_3plus': res_df['is_3plus'].mean(),
            'conditional_3plus': res_df[res_df['k'] > 0]['is_3plus'].mean() if active_days > 0 else 0,
            'unconditional_1plus': res_df['is_1plus'].mean(),
            'avg_5d_return': res_df[res_df['k'] > 0]['avg_return'].mean() if active_days > 0 else 0,
            'worst_day': res_df['avg_return'].min(),
            'median_hits': res_df[res_df['k'] > 0]['hits'].median() if active_days > 0 else 0
        }

        # Regime Breakdown
        if regime_detector:
            regime_stats = res_df.groupby('regime').agg({
                'is_3plus': 'mean',
                'avg_return': 'mean',
                'k': 'count'
            }).rename(columns={'k': 'days'})
            metrics['regime_breakdown'] = regime_stats.to_dict('index')

        return metrics, res_df
