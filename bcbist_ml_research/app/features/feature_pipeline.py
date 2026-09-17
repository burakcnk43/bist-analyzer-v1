import os
import pandas as pd
import logging
from typing import Dict, List
from joblib import Parallel, delayed
from app.features.technical import add_technical_features
from app.features.volume import add_volume_features
from app.features.sector_features import add_sector_features
from app.features.event_features import add_event_features
from app.features.relative import add_relative_features
from app.features.interactions import add_elite_interactions
from app.labels.target_builder import add_targets, add_cross_sectional_targets
from app.config import STOCK_UNIVERSE
from app.data.event_data import get_event_dataset

logger = logging.getLogger(__name__)

def process_single_symbol(symbol, df, include_events, events_df):
    """
    Worker function for parallel processing.
    """
    try:
        # 1. Base features
        df = add_technical_features(df)
        df = add_volume_features(df)
        if include_events:
            df = add_event_features(df, events_df, symbol)

        # Build raw returns targets
        df = add_targets(df)
        df['symbol'] = symbol
        return symbol, df
    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")
        return symbol, None

def run_feature_pipeline(all_data: Dict[str, pd.DataFrame], include_events=True) -> Dict[str, pd.DataFrame]:
    processed_data = {}

    # 0. Events Data
    events_df = pd.DataFrame()
    if include_events:
        symbols = list(all_data.keys())
        # Optimization: use a subset of dates for event fetching to avoid heavy concat
        # We only need start/end
        dates = []
        for s in list(all_data.keys())[:5]: # Sample 5 to get range
            dates.extend([all_data[s].index.min(), all_data[s].index.max()])
        start_date = min(dates).strftime("%Y-%m-%d")
        end_date = max(dates).strftime("%Y-%m-%d")
        events_df = get_event_dataset(symbols, start_date, end_date)

    # 1. Base Logic
    is_android = "ANDROID_ROOT" in os.environ
    n_workers = 1 if is_android else -1

    logger.info(f"Starting feature building (workers={n_workers}) for {len(all_data)} symbols...")
    results = Parallel(n_jobs=n_workers)(
        delayed(process_single_symbol)(symbol, df, include_events, events_df)
        for symbol, df in all_data.items()
    )

    global_list = []
    for symbol, df in results:
        if df is not None:
            processed_data[symbol] = df
            global_list.append(df)

    # 2. Sector and Market-Relative Logic (Cross-Sectional)
    sector_mapping = {s['symbol']: s.get('sector', 'Other') for s in STOCK_UNIVERSE}

    global_df = pd.concat(global_list).sort_index()
    # Add sector column
    global_df['sector'] = global_df['symbol'].map(sector_mapping)

    logger.info("Building cross-sectional relative features...")
    global_df = add_relative_features(global_df)

    logger.info("Building cross-sectional targets...")
    global_df = add_cross_sectional_targets(global_df)

    logger.info("Generating high-order elite interactions...")
    global_df = add_elite_interactions(global_df)

    # 3. Final Split
    final_data = {}
    for symbol in processed_data.keys():
        final_data[symbol] = global_df[global_df['symbol'] == symbol].copy()

    return final_data
