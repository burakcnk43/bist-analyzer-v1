import pandas as pd
import numpy as np
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class IntradayGater:
    """
    Accuracy Booster: Confirms EOD predictions using 10:00-10:30 opening market data.
    Vetoes candidates that show weakness in price trajectory or volume intensity.
    """
    def __init__(self, min_intensity=0.8, min_trajectory=1.001):
        self.min_intensity = min_intensity
        self.min_trajectory = min_trajectory

    def evaluate_confirmation(self,
                               symbol: str,
                               intraday_df: pd.DataFrame) -> Dict[str, bool]:
        """
        Returns {'confirmed': True/False, 'reason': str}
        """
        if intraday_df.empty:
            return {'confirmed': True, 'reason': "No intraday data (Defaulting to EOD)"}

        # yfinance period='1d' returns current day bars
        if len(intraday_df) < 1:
            return {'confirmed': True, 'reason': "Market not open yet or data lag"}

        # 1. Price Trajectory (Open vs Latest)
        opening_price = intraday_df['Open'].iloc[0]
        current_price = intraday_df['Close'].iloc[-1]
        trajectory = current_price / opening_price

        # 2. Volume Intensity
        avg_vol = intraday_df['Volume'].mean()
        current_vol = intraday_df['Volume'].iloc[-1]
        intensity = current_vol / (avg_vol + 1e-9)

        confirmed = True
        reason = "Intraday momentum confirmed"

        # Veto if price dropped significantly below open
        if trajectory < 0.995:
            confirmed = False
            reason = f"Intraday price drop: {trajectory:.2%} vs Open"
        # Veto if volume is extremely thin (less than 30% of opening average)
        elif intensity < 0.3 and len(intraday_df) > 3:
            confirmed = False
            reason = f"Thin opening volume: {intensity:.2f}x intensity"

        return {'confirmed': confirmed, 'reason': reason}

    def filter_candidates(self, candidates: pd.DataFrame, intraday_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Applies confirmation logic to the Top-K pool.
        """
        results = []
        for _, row in candidates.iterrows():
            symbol = row['symbol_col']
            df_intra = intraday_data.get(symbol, pd.DataFrame())

            conf = self.evaluate_confirmation(symbol, df_intra)

            if conf['confirmed']:
                results.append(row)
            else:
                logger.info(f"Vetoed {symbol} due to intraday check: {conf['reason']}")

        return pd.DataFrame(results)
