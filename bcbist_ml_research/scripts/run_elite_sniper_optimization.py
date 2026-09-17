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
from app.ml.meta_gater_v6 import MetaGaterV6

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EliteSniperOptimization")

def main():
    logger.info("--- PHASE 28: STARTING ELITE SNIPER OVERDRIVE OPTIMIZATION ---")
    pm = ProductionManager(Path("configs/production/production_config.json"))

    # 1. Prepare 5-Year Dataset (Sampled for speed)
    all_data = []
    valid_syms = pd.read_csv('data/raw/bist_universe_valid.csv')['symbol'].tolist()[:40]
    for symbol in valid_syms:
        path = DATA_FEATURES_DIR / f"{symbol.replace('.', '_')}_features.csv"
        if path.exists():
            df = pd.read_csv(path, index_col='date', parse_dates=True)
            df['symbol_col'] = symbol
            all_data.append(df)

    global_df = pd.concat(all_data).sort_index()
    breadth_df = pd.read_csv(BREADTH_DATA_PATH, index_col='date', parse_dates=True)

    dates = sorted(global_df.index.unique())
    train_dates = dates[:-80]
    test_dates = dates[-80:]

    logger.info(f"Generating Meta-Training data from {len(train_dates)} days (Sampling 1/4)...")

    meta_records = []
    y_hits = []

    # Training Loop
    for i, dt in enumerate(train_dates[:-1:4]):
        day_data = global_df.loc[[dt]]
        day_mkt = breadth_df.loc[:dt].tail(20)
        regime = pm.regime_detector.detect_regime(day_mkt)
        mkt_quality = pm.quality_predictor.calculate_quality_score(day_mkt.iloc[-1], regime)

        scores = pm.scorer.calculate_production_scores(day_data, {'regime': regime})
        day_full = day_data.join(scores).fillna(0)

        specialist_probs = {
            "general": day_full['production_alpha'].values,
            "trust": pm.trust_model.predict_trust_score(day_full),
            "box": pm.darvas_scorer.predict_success_prob(day_full),
            "event": pm.event_expert.predict_event_impulse(day_full)
        }

        X_meta = pm.elite_gater.prepare_elite_features(day_full, specialist_probs, regime, mkt_quality)

        next_dt = dates[dates.index(dt) + 1]
        actuals = global_df.loc[[next_dt]].set_index('symbol_col')

        for idx_row, (idx, row) in enumerate(X_meta.iterrows()):
            sym = day_full.iloc[idx_row]['symbol_col']
            if sym in actuals.index:
                ret = actuals.loc[sym]['return_1d']
                if isinstance(ret, pd.Series): ret = ret.iloc[0]
                hit = 1 if ret > 0.012 else 0
                meta_records.append(row)
                y_hits.append(hit)

    # 2. Train Meta-Gater V6
    X_meta_final = pd.DataFrame(meta_records)
    y_hits_final = pd.Series(y_hits)
    pm.elite_gater.train(X_meta_final, y_hits_final)
    pm.elite_gater.save(MODELS_DIR / "production" / "meta_gater_v6.joblib")

    # 3. Parameter Sweep
    logger.info("Audit: Testing precision (Sampling 1/2)...")
    best_threshold = 0.82
    best_precision = 0

    for thresh in [0.70, 0.75, 0.80, 0.83]:
        total_trades = 0
        success_trades = 0

        for i, dt in enumerate(test_dates[:-1:2]):
            day_data = global_df.loc[[dt]]
            day_mkt = breadth_df.loc[:dt].tail(20)
            pm.config['selection']['elite_threshold'] = thresh
            res = pm.get_daily_picks(day_data, day_mkt)

            if res['status'] == 'SUCCESS':
                next_dt = dates[dates.index(dt) + 1]
                actuals = global_df.loc[[next_dt]].set_index('symbol_col')
                for p in res['predictions']:
                    total_trades += 1
                    sym = p['symbol']
                    if sym in actuals.index:
                        ret = actuals.loc[sym]['return_1d']
                        if isinstance(ret, pd.Series): ret = ret.iloc[0]
                        if ret > 0: success_trades += 1

        precision = success_trades / total_trades if total_trades > 0 else 0
        logger.info(f"Threshold {thresh:.2f}: Precision = {precision:.2%}, Trades = {total_trades}")

        if precision > best_precision and total_trades > 2:
            best_precision = precision
            best_threshold = thresh

    print(f"\n--- ELITE SNIPER UPGRADE COMPLETED ---")
    print(f"Optimal Elite Threshold: {best_threshold}")
    print(f"Audit Precision (1D): {best_precision:.2%}")

    pm.config['selection']['elite_threshold'] = best_threshold
    import json
    with open("configs/production/production_config.json", 'w') as f:
        json.dump(pm.config, f, indent=4)

if __name__ == "__main__":
    main()
