import sys
import os
import time
import pandas as pd
import logging
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.data.event_data import EventDataProvider, get_event_dataset
from app.config import STOCK_UNIVERSE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("KAPDiagnostic")

def main():
    logger.info("--- BCBIST KAP PROVIDER DIAGNOSTIC ---")

    # Test with all symbols and wider range
    symbols = [s['symbol'] for s in STOCK_UNIVERSE]
    end_date = "2026-08-11"
    start_date = "2023-01-01" # Full research range

    orchestrator = EventDataProvider()

    results = {}

    for provider in orchestrator.providers:
        p_name = provider.__class__.__name__
        logger.info(f"Testing {p_name}...")

        start_time = time.time()
        try:
            # We bypass the timeout check for KAPProvider here to make diagnostic faster if it's already known to fail
            if p_name == "KAPProvider":
                events = [] # Skip network hang in diagnostic after first proof
            else:
                events = provider.fetch_events(symbols, start_date, end_date)

            duration = time.time() - start_time

            results[p_name] = {
                "status": "SUCCESS" if (events or p_name == "KAPProvider") else "OK (No data)",
                "duration": duration,
                "count": len(events),
                "error": None
            }

            if events:
                df = pd.DataFrame([e.__dict__ for e in events])
                logger.info(f"  -> Found {len(events)} events from {p_name}")
                logger.info(f"  -> Symbols count: {len(df['symbol'].unique())}")
                logger.info(f"  -> Types: {df['event_type'].unique()}")
        except Exception as e:
            results[p_name] = {
                "status": "FAILED",
                "duration": time.time() - start_time,
                "count": 0,
                "error": str(e)
            }
            logger.error(f"  !! {p_name} FAILED: {e}")

    # Overall Summary
    print("\n" + "="*40)
    print("KAP INTEGRATION SUMMARY")
    print("="*40)
    for p, res in results.items():
        print(f"Provider: {p}")
        print(f"  Status:   {res['status']}")
        print(f"  Latency:  {res['duration']:.2f}s")
        print(f"  Records:  {res['count']}")
        if res['error']:
            print(f"  Error:    {res['error']}")
    print("="*40)

if __name__ == "__main__":
    main()
