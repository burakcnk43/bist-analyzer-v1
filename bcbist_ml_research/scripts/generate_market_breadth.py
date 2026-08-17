import sys
import os
import pandas as pd
import logging
from joblib import Parallel, delayed

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.features.breadth import MarketBreadthEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_and_prepare(filepath, engine):
    """
    Loads a feature file and prepares it for breadth calculation.
    """
    try:
        df = pd.read_csv(filepath)
        if 'date' not in df.columns or 'symbol' not in df.columns:
            return None

        # Ensure required base columns are present for indicator calculation
        base_cols = ['date', 'symbol', 'close', 'high', 'low', 'sma_20', 'sma_50', 'sma_200', 'rsi_14']
        missing = [c for c in base_cols if c not in df.columns]
        if any(c in missing for c in ['close', 'high', 'low']):
            return None

        # Apply per-symbol indicators in parallel using the engine's helper
        df = engine.prepare_indicators(df)

        # Keep only necessary columns for aggregation to save memory
        final_cols = [
            'date', 'symbol', 'is_breakout_close', 'is_breakout_down',
            'is_advancing', 'is_declining', 'is_new_high_20d', 'is_new_low_20d',
            'is_above_sma20', 'is_above_sma50', 'is_above_sma200',
            'is_rsi_above_50', 'is_rsi_below_30',
            'is_pos_1d', 'is_pos_3d', 'is_pos_5d', 'is_macd_pos',
            'is_vol_exp', 'is_rel_strength_pos'
        ]
        available = [c for c in final_cols if c in df.columns]
        return df[available]

    except Exception as e:
        logger.error(f"Error processing {filepath}: {e}")
        return None

def main():
    features_dir = "data/features"
    output_path = "data/market_breadth.csv"

    if not os.path.exists(features_dir):
        logger.error(f"Features directory not found: {features_dir}")
        return

    files = [os.path.join(features_dir, f) for f in os.listdir(features_dir) if f.endswith("_features.csv")]

    if not files:
        logger.error(f"No feature files found in {features_dir}")
        return

    logger.info(f"Processing {len(files)} symbols in parallel...")

    engine = MarketBreadthEngine()

    # Parallel loading and per-symbol calculation
    results = Parallel(n_jobs=-1)(
        delayed(load_and_prepare)(f, engine) for f in files
    )

    # Filter out None and concatenate
    dfs = [df for df in results if df is not None]

    if not dfs:
        logger.error("No valid symbol data loaded.")
        return

    universe_data = pd.concat(dfs, ignore_index=True)
    logger.info(f"Concatenated processed data: {universe_data.shape[0]} rows.")

    # Calculate final breadth
    logger.info("Aggregating daily market breadth metrics...")
    breadth_df = engine.calculate_breadth(universe_data)

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    breadth_df.to_csv(output_path)

    logger.info(f"Success! Market breadth saved to {output_path}")
    if not breadth_df.empty:
        logger.info(f"Date range: {breadth_df.index.min()} to {breadth_df.index.max()}")
        logger.info(f"Metrics generated: {list(breadth_df.columns)}")

if __name__ == "__main__":
    main()
