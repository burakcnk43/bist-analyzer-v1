import sys
import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import DATA_FEATURES_DIR, BREADTH_DATA_PATH, MODELS_DIR
from app.ml.production_manager import ProductionManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SelfImprovement")

def main():
    logger.info("--- PHASE 27: STARTING HISTORICAL SELF-OPTIMIZATION WARMUP ---")
    pm = ProductionManager(Path("configs/production/production_config.json"))

    # 1. Prepare Historical Pool
    all_data = []
    # Using Top 30 symbols for quick optimization
    valid_syms = pd.read_csv('data/raw/bist_universe_valid.csv')['symbol'].tolist()[:30]
    for symbol in valid_syms:
        path = DATA_FEATURES_DIR / f"{symbol.replace('.', '_')}_features.csv"
        if path.exists():
            df = pd.read_csv(path, index_col='date', parse_dates=True)
            df['symbol_col'] = symbol
            all_data.append(df)

    global_df = pd.concat(all_data).sort_index()
    breadth_df = pd.read_csv(BREADTH_DATA_PATH, index_col='date', parse_dates=True)

    dates = sorted(global_df.index.unique())
    # Optimized over the last 150 trading days (approx 7 months)
    test_dates = dates[-150:]

    logger.info(f"Simulating {len(test_dates)} days of learning...")

    for i, current_date in enumerate(test_dates):
        # A. Feedback Step: Realize 1D outcome for T-1
        if i > 0:
            prev_date = test_dates[i-1]
            picks = pm.memory.prediction_log
            if not picks.empty:
                pending = picks[picks['outcome_class'] == 'PENDING']
                outcomes = []
                for _, p in pending.iterrows():
                    sym = p['symbol']
                    # Look ahead safely for the optimizer loop to 'teach' the model
                    if prev_date in global_df.index:
                        # actual_return_1d
                        row = global_df.loc[prev_date]
                        if isinstance(row, pd.DataFrame):
                            row = row[row['symbol_col'] == sym]
                            ret = row['return_1d'].iloc[0] if not row.empty else 0
                        else:
                            ret = row['return_1d'] if row['symbol_col'] == sym else 0

                        outcomes.append({'date': p['date'], 'symbol': sym, 'actual_return_1d': ret})

                if outcomes:
                    pm.memory.update_outcomes(pd.DataFrame(outcomes), horizon='1d')
                    # Update MetaLearner weights based on new hit/miss
                    for o in outcomes:
                        pm.meta_learner.record_outcome("general", pd.to_datetime(o['date']), o['actual_return_1d'] > 0)

        # B. Prediction Step (T)
        day_data = global_df.loc[[current_date]]
        mkt_data = breadth_df.loc[:current_date].tail(20)

        # This will automatically log to memory and generate PDF
        pm.get_daily_picks(day_data, mkt_data)

        if i % 30 == 0:
            logger.info(f"Day {i}/{len(test_dates)}: Rolling Accuracy = {pm.memory.get_recent_performance().get('hit_rate', 0):.2%}")

    # Save final optimized state
    prod_models_dir = MODELS_DIR / "production"
    pm.meta_learner.save_state(prod_models_dir / "meta_learner_state.joblib")
    pm.memory.save_memory()

    final_perf = pm.memory.get_recent_performance(days=150)
    print("\n--- WARMUP COMPLETED ---")
    print(f"Final Optimized Hit Rate (1D): {final_perf.get('hit_rate', 0):.2%}")
    print(f"Total Learning Cycles: {len(test_dates)}")
    print("Optimized meta-weights saved to production artifacts.")

if __name__ == "__main__":
    main()
