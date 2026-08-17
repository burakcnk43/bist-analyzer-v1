import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import DATA_FEATURES_DIR

def main():
    print("--- PHASE 5: TARGET INTEGRITY AUDIT ---")

    # Load a sample feature file
    file_path = list(DATA_FEATURES_DIR.glob("THYAO_IS_features.csv"))[0]
    df = pd.read_csv(file_path, index_col='date', parse_dates=True)

    # Random sample of dates
    sample_indices = [100, 500, 1000]

    audit_data = []

    for idx in sample_indices:
        if idx >= len(df) - 20: continue

        current_date = df.index[idx]
        current_close = df.iloc[idx]['close']

        future_20_date = df.index[idx+20]
        future_20_close = df.iloc[idx+20]['close']

        target_val = df.iloc[idx]['target_up_20d']
        calc_ret = (future_20_close / current_close) - 1
        calc_target = 1 if calc_ret > 0 else 0

        match = (int(target_val) == int(calc_target))

        print(f"Date: {current_date.date()}")
        print(f"  Close(T):      {current_close:.2f}")
        print(f"  Date(T+20):    {future_20_date.date()}")
        print(f"  Close(T+20):    {future_20_close:.2f}")
        print(f"  Target in CSV: {target_val}")
        print(f"  Calc Return:   {calc_ret:.4%}")
        print(f"  Match:         {match}")
        print("-" * 30)

        audit_data.append({
            "date": current_date,
            "target": target_val,
            "calc_ret": calc_ret,
            "match": match
        })

    audit_df = pd.DataFrame(audit_data)
    audit_df.to_csv("PHASE5_TARGET_AUDIT.csv", index=False)

if __name__ == "__main__":
    main()
