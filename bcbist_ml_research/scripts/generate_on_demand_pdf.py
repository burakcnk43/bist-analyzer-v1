import sys
import os
import pandas as pd
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import DATA_FEATURES_DIR, BREADTH_DATA_PATH, MODELS_DIR
from app.ml.production_manager import ProductionManager

def main():
    pm = ProductionManager(Path("configs/production/production_config.json"))

    # 1. Discover date
    target_date = sys.argv[1] if len(sys.argv) > 1 else None

    if not target_date:
        for f in DATA_FEATURES_DIR.glob("*.csv"):
            with open(f, 'r') as file:
                lines = file.readlines()
                if len(lines) > 1:
                    target_date = lines[-1].split(',')[0]
                    break

    if not target_date:
        print("Error: No date found.")
        return

    print(f"Generating PDF report for: {target_date}")

    # 2. Load context
    all_dfs = []
    for f in DATA_FEATURES_DIR.glob("*.csv"):
        df = pd.read_csv(f, index_col='date', parse_dates=True)
        if target_date in df.index:
            row = df.loc[[target_date]].copy()
            row['symbol_col'] = f.stem.replace('_features', '').replace('_', '.')
            all_dfs.append(row)

    if not all_dfs:
        print(f"No data found for {target_date}")
        return

    day_data = pd.concat(all_dfs)
    mkt_data = pd.read_csv(BREADTH_DATA_PATH, index_col='date', parse_dates=True).loc[:target_date].tail(20)

    # 3. Generate Picks & PDF
    response = pm.get_daily_picks(day_data, mkt_data)

    print("\nSUCCESS: PDF Generated.")
    print(f"Market Regime: {response['market_regime']}")
    print(f"Selections: {len(response['predictions'])}")

if __name__ == "__main__":
    main()
