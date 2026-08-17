import sys
import os
import requests
import pandas as pd
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import DATA_RAW_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("UniverseDiscovery")

def main():
    logger.info("Starting BIST Universe Discovery...")

    # 1. Fetch from GitHub (ahmeterenodaci repo - high coverage)
    url = "https://raw.githubusercontent.com/ahmeterenodaci/Istanbul-Stock-Exchange--BIST--including-symbols-and-logos/master/bist.json"

    try:
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()

        symbols_list = []
        for item in data:
            sym = item['symbol'].strip()
            # Clean symbols (some might have weird chars)
            # Add .IS suffix for yfinance if not present
            if not sym.endswith(".IS"):
                sym = f"{sym}.IS"

            symbols_list.append({
                "symbol": sym,
                "name": item.get('name', 'Unknown'),
                "sector": "Other" # We'll need to enrich this or use default
            })

        df = pd.DataFrame(symbols_list)

        # Deduplicate
        df = df.drop_duplicates(subset=['symbol'])

        # 2. Save to Raw Data
        path = DATA_RAW_DIR / "bist_universe.csv"
        df.to_csv(path, index=False)

        logger.info(f"Discovered {len(df)} symbols. Saved to {path}")

        # 3. Create Audit Template
        audit_path = DATA_RAW_DIR / "PHASE7_UNIVERSE_AUDIT.csv"
        df.to_csv(audit_path, index=False)

    except Exception as e:
        logger.error(f"Failed to discover universe: {e}")

if __name__ == "__main__":
    main()
