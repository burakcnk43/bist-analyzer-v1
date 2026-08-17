import sys
import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from itertools import combinations

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import DATA_FEATURES_DIR, DATA_REPORTS_DIR, STOCK_UNIVERSE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("InteractionMiner")

def main():
    logger.info("--- PHASE 12: NON-LINEAR FEATURE INTERACTION MINING ---")

    # 1. Sample Data
    all_data = []
    # Use top 20 symbols for faster mining
    valid_syms = pd.read_csv('data/raw/bist_universe_valid.csv')['symbol'].tolist()
    for symbol in valid_syms[:20]:
        path = DATA_FEATURES_DIR / f"{symbol.replace('.', '_')}_features.csv"
        if path.exists():
            df = pd.read_csv(path, index_col='date', parse_dates=True)
            all_data.append(df)

    df = pd.concat(all_data).dropna(subset=['target_return_5d'])

    core_features = ['rsi_14', 'bb_pct', 'relative_volume', 'dist_sma_20', 'macdh_12_26_9']
    target = 'target_return_5d'

    interactions = []

    # 2. Brute Force Product Interactions
    for f1, f2 in combinations(core_features, 2):
        # Calculate product
        inter_name = f"{f1}_x_{f2}"
        inter_val = df[f1] * df[f2]

        # Measure correlation with target
        corr = np.corrcoef(inter_val, df[target])[0, 1]

        # Baseline corrs
        c1 = np.corrcoef(df[f1], df[target])[0, 1]
        c2 = np.corrcoef(df[f2], df[target])[0, 1]

        if abs(corr) > max(abs(c1), abs(c2)) + 0.01:
            interactions.append({
                "interaction": inter_name,
                "corr": corr,
                "improvement": abs(corr) - max(abs(c1), abs(c2))
            })

    if not interactions:
        logger.warning("No significant interactions found with current threshold.")
        inter_df = pd.DataFrame(columns=["interaction", "corr", "improvement"])
    else:
        inter_df = pd.DataFrame(interactions).sort_values('improvement', ascending=False)
    inter_df.to_csv(DATA_REPORTS_DIR / "PHASE12_INTERACTIONS.csv", index=False)

    # 3. Report
    report = "# Phase 12: Feature Interaction Discoveries\n\n"
    report += "Identified nonlinear interactions with higher predictive power than individual components.\n\n"
    report += inter_df.head(10).to_markdown(index=False)

    with open(DATA_REPORTS_DIR / "PHASE12_INTERACTION_REPORT.md", "w") as f:
        f.write(report)

    logger.info(f"Interaction mining complete. Found {len(inter_df)} significant interactions.")

if __name__ == "__main__":
    main()
