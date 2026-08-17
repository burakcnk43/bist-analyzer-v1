import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class EconomicBacktester:
    def __init__(self, transaction_cost=0.001): # 10 bps default
        self.transaction_cost = transaction_cost

    def run_simple_backtest(self, test_df: pd.DataFrame, y_prob: np.ndarray, horizon=1):
        """
        Simulates a long-only strategy based on top model predictions.
        """
        df = test_df.copy()
        df['prob'] = y_prob

        # Rank by probability for each day
        # We assume we buy the top 5 predicted stocks each day
        daily_probs = df.groupby(level=0)['prob'].nlargest(5)

        # Merge with returns
        # Forward return is for the next 'horizon' days
        ret_col = f'target_return_{horizon}d'
        if ret_col not in df.columns:
            logger.error(f"Return column {ret_col} not found for backtest.")
            return {}

        # Get returns for those top 5
        # Index of daily_probs is (date, original_index)
        top_indices = daily_probs.index.get_level_values(1)
        top_returns = df.loc[top_indices, ret_col]

        # Average daily return of top 5
        strategy_returns = top_returns.groupby(level=0).mean()

        # Adjust for transaction costs (once per horizon period)
        adj_returns = strategy_returns - self.transaction_cost

        cumulative = (1 + adj_returns).cumprod()

        # Performance Metrics
        total_ret = cumulative.iloc[-1] - 1 if not cumulative.empty else 0
        sharpe = (adj_returns.mean() / adj_returns.std() * np.sqrt(252)) if len(adj_returns) > 1 else 0

        # Drawdown
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = drawdown.min()

        return {
            "cumulative_return": total_ret,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "win_rate": (adj_returns > 0).mean()
        }
