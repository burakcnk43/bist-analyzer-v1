import sys
import os
import pandas as pd
import logging
from typing import Dict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import STOCK_UNIVERSE, DATA_RAW_DIR, DATA_FEATURES_DIR
from app.features.feature_pipeline import run_feature_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FeatureBuilding")

def normalize_index(df):
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        return df.tz_localize(None)
    # Check if first element is Timestamp with tz
    if not df.empty and isinstance(df.index[0], pd.Timestamp) and df.index[0].tzinfo is not None:
        df.index = [d.replace(tzinfo=None) for d in df.index]
    return df

def main():
    logger.info("Starting Feature Engineering Pipeline")

    # 1. Load Raw Market Data
    all_raw_data = {}
    for entry in STOCK_UNIVERSE:
        symbol = entry['symbol']
        safe_name = symbol.replace('.', '_')
        path = DATA_RAW_DIR / f"{safe_name}.csv"

        if os.path.exists(path):
            df = pd.read_csv(path, index_col='date', parse_dates=True)
            df = normalize_index(df)
            all_raw_data[symbol] = df
            logger.info(f"Loaded {symbol} ({len(df)} rows)")

    if not all_raw_data:
        logger.error("No raw data found. Run collect_data.py first.")
        return

    # 2. Add Macro Features
    logger.info("Joining Macro Features...")
    macro_df = pd.DataFrame()
    for macro in ['usdtry', 'eurtry', 'bist100', 'gold', 'brent', 'sp500']:
        path = DATA_RAW_DIR / f"macro_{macro}.csv"
        if os.path.exists(path):
            mdf = pd.read_csv(path, index_col='date', parse_dates=True)
            mdf = normalize_index(mdf)

            series = mdf['close'].rename(f"macro_{macro}_close")
            ret = mdf['close'].pct_change().rename(f"macro_{macro}_return")

            if macro_df.empty:
                macro_df = pd.concat([series, ret], axis=1)
            else:
                macro_df = macro_df.join(pd.concat([series, ret], axis=1), how='outer')

    macro_df = macro_df.ffill()

    for symbol in all_raw_data:
        all_raw_data[symbol] = all_raw_data[symbol].join(macro_df, how='left').ffill()

    # 3. Run Pipeline
    logger.info("Running Feature Engineering...")
    final_dataset = run_feature_pipeline(all_raw_data)

    # 4. Save Features
    feature_counts = {}
    for symbol, df in final_dataset.items():
        safe_name = symbol.replace('.', '_')
        path = DATA_FEATURES_DIR / f"{safe_name}_features.csv"
        df.to_csv(path)
        feature_counts[symbol] = df.shape[1]

    logger.info("--- FEATURE BUILDING SUMMARY ---")
    logger.info(f"Symbols Processed: {len(final_dataset)}")
    logger.info(f"Avg Features per Symbol: {sum(feature_counts.values()) / len(feature_counts)}")
    logger.info(f"Feature datasets saved to {DATA_FEATURES_DIR}")

if __name__ == "__main__":
    main()
