import pandas as pd
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.config import DATA_FEATURES_DIR, STOCK_UNIVERSE
from app.features.sector_rotation import SectorRotationEngine, get_sector_mapping

def main():
    print("Initializing Sector Rotation Engine...")
    engine = SectorRotationEngine()

    # Get sector mapping from config as backup/override
    sector_map = get_sector_mapping(STOCK_UNIVERSE)

    all_data = []
    feature_files = list(DATA_FEATURES_DIR.glob("*_features.csv"))

    if not feature_files:
        print(f"No feature files found in {DATA_FEATURES_DIR}")
        return

    print(f"Loading {len(feature_files)} feature files...")

    for i, file_path in enumerate(feature_files):
        if i % 100 == 0:
            print(f"Processing file {i}/{len(feature_files)}: {file_path.name}")
        try:
            # Only read necessary columns to save memory
            df = pd.read_csv(file_path, usecols=[
                'date', 'symbol', 'sector', 'return_1d', 'return_5d',
                'relative_volume', 'dist_20d_high', 'rsi_14', 'rolling_std_20'
            ])

            # If sector is missing or 'Other', try to use mapping from config
            if sector_map:
                df['sector_config'] = df['symbol'].map(sector_map)
                df['sector'] = df['sector_config'].fillna(df['sector'])
                df.drop(columns=['sector_config'], inplace=True)

            all_data.append(df)
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    if not all_data:
        print("No data collected.")
        return

    print("Concatenating data...")
    universe_df = pd.concat(all_data, ignore_index=True)

    print("Calculating sector statistics...")
    sector_rotation_df = engine.calculate_sector_stats(universe_df)

    output_path = Path(__file__).resolve().parent.parent / "data" / "sector_rotation.csv"
    print(f"Saving sector rotation data to {output_path}...")
    sector_rotation_df.to_csv(output_path, index=False)

    print("Done!")

if __name__ == "__main__":
    main()
