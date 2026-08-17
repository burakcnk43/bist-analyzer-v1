import sys
import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import DATA_REPORTS_DIR
from app.ml.memory import MemoryEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MemoryInit")

def main():
    logger.info("Initializing Success Memory from historical logs...")

    memory = MemoryEngine(DATA_REPORTS_DIR / "PHASE11_SUCCESS_MEMORY.csv")

    # 1. Ingest Phase 8 Log (The big one)
    p8_path = DATA_REPORTS_DIR / "PHASE8_PREDICTION_LOG.csv"
    if p8_path.exists():
        logger.info("Loading Phase 8 Log...")
        # Note: This is large, we might want to sample or filter for top 20
        p8_df = pd.read_csv(p8_path)
        p8_df['date'] = pd.to_datetime(p8_df['date'])

        # Prepare for MemoryEngine structure
        mem_df = pd.DataFrame({
            "date": p8_df['date'],
            "symbol": p8_df.get('symbol', p8_df.get('symbol_col')),
            "sector": p8_df.get('sector', 'Other'),
            "rank": p8_df.get('daily_rank', 0),
            "raw_score": p8_df.get('alpha_score', 0),
            "actual_return_5d": p8_df.get('target_return_5d', 0)
        })

        # Filter for top predictions to keep memory focused on what we care about
        mem_df = mem_df[mem_df['rank'] <= 20]

        memory.add_predictions(mem_df)
        logger.info(f"Added {len(mem_df)} top-20 predictions from Phase 8.")

    # 2. Ingest Phase 10 Fresh Results
    p10_path = DATA_REPORTS_DIR / "PHASE10_FRESH_PREDICTIONS.csv"
    p10_res_path = DATA_REPORTS_DIR / "PHASE10_FRESH_RESULTS.csv"

    if p10_path.exists() and p10_res_path.exists():
        logger.info("Loading Phase 10 Fresh Data...")
        p10_df = pd.read_csv(p10_path)
        p10_df = p10_df[p10_df['model'] == 'PHASE10']

        res_df = pd.read_csv(p10_res_path) # Daily summaries
        # We need per-symbol outcomes, let's grab from features if needed
        # But for now let's just initialize the basic structure

        mem_p10 = pd.DataFrame({
            "date": pd.to_datetime(p10_df['date']),
            "symbol": p10_df['symbol'],
            "sector": 'Unknown', # Will be joined later
            "rank": p10_df['rank'],
            "raw_score": p10_df['alpha'],
            "prob_5d": p10_df['prob_5d'],
            "regime": p10_df['regime']
        })
        memory.add_predictions(mem_p10)

    # 3. Finalize
    memory.update_outcomes(pd.DataFrame()) # Trigger classification
    memory.save_memory()

    logger.info("Memory initialization complete.")

if __name__ == "__main__":
    main()
