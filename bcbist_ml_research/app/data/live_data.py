import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from app.config import STOCK_UNIVERSE, DATA_RAW_DIR, DATA_FEATURES_DIR
from app.data.market_data import MarketDataProvider
from app.features.feature_pipeline import run_feature_pipeline

logger = logging.getLogger(__name__)

class LiveDataProvider:
    """
    Orchestrates real-time data ingestion and feature building for production.
    """
    def __init__(self):
        self.market_provider = MarketDataProvider()

    def get_latest_production_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
        """
        1. Discovers universe.
        2. Fetches latest data.
        3. Builds features.
        4. Returns (day_data, market_data, latest_date).
        """
        logger.info("Starting live data ingestion...")

        # 1. Fetch latest prices for universe
        # For production speed, we might limit to top N or recently active symbols
        symbols = [s['symbol'] for s in STOCK_UNIVERSE]

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")

        all_raw = {}
        for sym in symbols[:150]: # Limited universe for production responsiveness
            df = self.market_provider.fetch_ohlcv(sym, start_date, end_date)
            if not df.empty:
                all_raw[sym] = df

        if not all_raw:
            raise ValueError("Failed to fetch any market data.")

        # 2. Build Features
        logger.info("Building live features...")
        # Need to join macro features here too (reusing logic from build_features.py)
        # (Simplified: assuming macro files are updated or using latest prices)

        processed_data = run_feature_pipeline(all_raw)

        # 3. Align to latest date
        dates = []
        for sym in processed_data:
            dates.append(processed_data[sym].index.max())

        latest_date = max(dates)
        logger.info(f"Latest common production date: {latest_date}")

        all_rows = []
        for sym, df in processed_data.items():
            if latest_date in df.index:
                row = df.loc[[latest_date]].copy()
                row['symbol_col'] = sym
                all_rows.append(row)

        day_data = pd.concat(all_rows)

        # 4. Market Breadth (Re-calculate or load latest)
        # For now, load from market_breadth.csv (which should be updated by a cron)
        mkt_df = pd.read_csv('data/market_breadth.csv', index_col='date', parse_dates=True)
        mkt_data = mkt_df[mkt_df.index <= latest_date].tail(20)

        return day_data, mkt_data, latest_date.strftime("%Y-%m-%d")
