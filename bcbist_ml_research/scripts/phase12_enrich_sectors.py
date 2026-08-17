import sys
import os
import pandas as pd
import logging
import yfinance as yf
from pathlib import Path
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import DATA_RAW_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SectorEnrichment")

def main():
    logger.info("Enriching BIST universe with real sector labels...")

    path = DATA_RAW_DIR / "bist_universe_valid.csv"
    if not path.exists():
        path = DATA_RAW_DIR / "bist_universe.csv"

    df = pd.read_csv(path)

    sector_map = {}

    # To save time in demo, we'll try to fetch for all but with a cache or limit if it takes too long.
    # In a real environment, we'd do all 495.

    for i, symbol in enumerate(df['symbol']):
        if i % 20 == 0:
            logger.info(f"Progress: {i}/{len(df)}")

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            sector = info.get('sector', 'Other')
            sector_map[symbol] = sector
            # Small sleep to be polite
            time.sleep(0.1)
        except Exception as e:
            logger.warning(f"Could not fetch sector for {symbol}: {e}")
            sector_map[symbol] = 'Other'

    df['sector'] = df['symbol'].map(sector_map)

    # Save enriched universe
    enriched_path = DATA_RAW_DIR / "bist_universe_enriched.csv"
    df.to_csv(enriched_path, index=False)

    # Update valid universe if it exists
    valid_path = DATA_RAW_DIR / "bist_universe_valid.csv"
    if valid_path.exists():
        df.to_csv(valid_path, index=False)

    logger.info(f"Sector enrichment complete. Summary:\n{df.sector.value_counts()}")

if __name__ == "__main__":
    main()
