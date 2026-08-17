import pandas as pd
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import DATA_FEATURES_DIR

def main():
    files = list(DATA_FEATURES_DIR.glob("*_features.csv"))
    if not files:
        print("No feature files found.")
        return

    # Load one as sample
    df = pd.read_csv(files[0], index_col=0)

    total_rows = len(df)

    groups = {
        "Technical": ['rsi_14', 'macd_12_26_9', 'bb_pct', 'sma_20'],
        "Volume": ['relative_volume', 'obv', 'pv_corr_20'],
        "Sector": ['sector_return_5d', 'relative_rsi_14'],
        "Macro": ['macro_usdtry_return', 'macro_bist100_return'],
        "KAP/Event": ['event_count_5d', 'days_since_last_event', 'contract_event_count_20d']
    }

    print(f"--- Feature Density Audit (File: {files[0].name}) ---")
    print(f"Total Rows: {total_rows}")
    print("-" * 50)
    print(f"{'Group':<15} | {'Feature':<25} | {'Fill Rate':<10}")
    print("-" * 50)

    for group, features in groups.items():
        for feat in features:
            if feat in df.columns:
                fill_rate = df[feat].notnull().mean() * 100
                print(f"{group:<15} | {feat:<25} | {fill_rate:>8.2f}%")
            else:
                print(f"{group:<15} | {feat:<25} | {'MISSING':>10}")

if __name__ == "__main__":
    main()
