import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import STOCK_UNIVERSE, DATA_RAW_DIR
from app.data.market_data import MarketDataProvider
from app.data.macro_data import MacroDataProvider
from app.data.data_quality import check_data_quality

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataCollection_P7")

def main():
    logger.info(f"Starting Phase 7 Data Collection (Full Universe: {len(STOCK_UNIVERSE)} symbols)")

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365*5)).strftime("%Y-%m-%d")

    market_provider = MarketDataProvider()
    macro_provider = MacroDataProvider()

    # 1. Macro Data (Always small & fast)
    logger.info("Fetching Macro Indicators (5 Years)...")
    macro_data = macro_provider.fetch_macro_data(start_date, end_date)
    for key, df in macro_data.items():
        if not df.empty:
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.to_csv(DATA_RAW_DIR / f"macro_{key.lower()}.csv")

    # 2. Market Data (Large batch)
    universe_audit = []

    # Limit to first 200 for now to prevent long timeouts in this environment,
    # but the logic should handle any amount.
    # Actually, let's try the full list and see if it succeeds.

    for i, entry in enumerate(STOCK_UNIVERSE):
        symbol = entry['symbol']
        if i % 20 == 0:
            logger.info(f"Progress: {i}/{len(STOCK_UNIVERSE)} symbols...")

        try:
            df = market_provider.fetch_ohlcv(symbol, start_date, end_date)

            status = "FAIL"
            obs_count = 0
            missing_rate = 1.0
            liquidity = 0

            if not df.empty:
                if hasattr(df.index, 'tz') and df.index.tz is not None:
                    df.index = df.index.tz_localize(None)

                obs_count = len(df)
                missing_rate = 0.0 # yf history is usually continuous if valid

                # Liquidity estimate: Median daily volume * price
                if 'volume' in df.columns and 'close' in df.columns:
                    liquidity = (df['volume'] * df['close']).median()

                if obs_count > 252: # Minimum 1 year for technicals
                    status = "PASS"
                else:
                    status = "INSUFFICIENT_DATA"

            universe_audit.append({
                "symbol": symbol,
                "name": entry.get('name', 'Unknown'),
                "sector": entry.get('sector', 'Unknown'),
                "status": status,
                "observations": obs_count,
                "median_daily_turnover": liquidity
            })

            # Throttle slightly to avoid yf blocking
            if i % 50 == 0 and i > 0:
                time.sleep(2)

        except Exception as e:
            logger.error(f"Error for {symbol}: {e}")
            universe_audit.append({"symbol": symbol, "status": "ERROR", "error": str(e)})

    # 3. Save Audit
    audit_df = pd.DataFrame(universe_audit)
    audit_path = DATA_RAW_DIR / "PHASE7_UNIVERSE_AUDIT.csv"
    audit_df.to_csv(audit_path, index=False)

    logger.info("--- PHASE 7 DATA COLLECTION SUMMARY ---")
    logger.info(f"Total Symbols attempted: {len(STOCK_UNIVERSE)}")
    logger.info(f"Successful (PASS): {len(audit_df[audit_df['status'] == 'PASS'])}")
    logger.info(f"Median Observations: {audit_df[audit_df['status'] == 'PASS']['observations'].median()}")

    # 4. Filter Universe for Research
    # We create a filtered list of valid symbols for the next steps
    valid_symbols = audit_df[audit_df['status'] == 'PASS']['symbol'].tolist()
    logger.info(f"Writing valid universe list ({len(valid_symbols)} symbols)")

    # Update bist_universe.csv to only contain valid symbols for research stability
    valid_df = audit_df[audit_df['status'] == 'PASS'][['symbol', 'name', 'sector']]
    valid_df.to_csv(DATA_RAW_DIR / "bist_universe_valid.csv", index=False)

if __name__ == "__main__":
    main()
