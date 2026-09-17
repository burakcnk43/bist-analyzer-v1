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
logger = logging.getLogger("DeepSniperSearch")

def main():
    logger.info("--- PHASE 28: STARTING DEEP SNIPER INTERACTION SEARCH ---")
    pm = ProductionManager(Path("configs/production/production_config.json"))

    valid_syms = pd.read_csv('data/raw/bist_universe_valid.csv')['symbol'].tolist()[:30]
    all_dfs = []
    for s in valid_syms:
        p = DATA_FEATURES_DIR / f"{s.replace('.', '_')}_features.csv"
        if p.exists():
            df = pd.read_csv(p, index_col='date', parse_dates=True)
            df['symbol_col'] = s
            all_dfs.append(df)

    full_df = pd.concat(all_dfs).sort_index()
    breadth_df = pd.read_csv(BREADTH_DATA_PATH, index_col='date', parse_dates=True)

    dates = sorted(full_df.index.unique())
    train_dates = dates[:-80]
    eval_dates = dates[-80:]

    logger.info(f"Training (Sampling 1/5)...")

    meta_X_train = []
    meta_y_train = []

    for dt in train_dates[:-1:5]:
        day_data = full_df.loc[[dt]]
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

        next_dt_idx = dates.index(dt) + 1
        if next_dt_idx < len(dates):
            next_dt = dates[next_dt_idx]
            actuals = full_df.loc[[next_dt]].set_index('symbol_col')

            for idx_row, (idx, row) in enumerate(X_meta.iterrows()):
                sym = day_full.iloc[idx_row]['symbol_col']
                if sym in actuals.index:
                    ret = actuals.loc[sym]['return_1d']
                    if isinstance(ret, pd.Series): ret = ret.iloc[0]
                    meta_X_train.append(row)
                    meta_y_train.append(1 if ret > 0.012 else 0)

    pm.elite_gater.train(pd.DataFrame(meta_X_train), pd.Series(meta_y_train))
    pm.elite_gater.save(MODELS_DIR / "production" / "meta_gater_v6.joblib")

    logger.info("Precision Tuning...")
    results = []
    for t in [0.75, 0.78, 0.80, 0.83]:
        correct = 0
        total = 0
        for dt in eval_dates[:-1:2]:
            day_data = full_df.loc[[dt]]
            day_mkt = breadth_df.loc[:dt].tail(20)
            pm.config['selection']['elite_threshold'] = t
            res = pm.get_daily_picks(day_data, day_mkt)
            if res['status'] == 'SUCCESS':
                next_dt = dates[dates.index(dt) + 1]
                actuals = full_df.loc[[next_dt]].set_index('symbol_col')
                for p in res['predictions']:
                    total += 1
                    sym = p['symbol']
                    if sym in actuals.index:
                        ret = actuals.loc[sym]['return_1d']
                        if isinstance(ret, pd.Series): ret = ret.iloc[0]
                        if ret > 0: correct += 1
        prec = correct / total if total > 0 else 0
        logger.info(f"Thresh {t:.2f}: Hit Rate = {prec:.2%}, Samples = {total}")
        results.append({'thresh': t, 'prec': prec, 'samples': total})

    filtered = [r for r in results if r['samples'] >= 1]
    best = max(filtered, key=lambda x: x['prec']) if filtered else results[0]

    print(f"\n--- SEARCH COMPLETED ---")
    print(f"Optimal Threshold: {best['thresh']:.2f}")
    print(f"Max Hit Rate: {best['prec']:.2%}")

    pm.config['selection']['elite_threshold'] = float(best['thresh'])
    import json
    with open("configs/production/production_config.json", 'w') as f:
        json.dump(pm.config, f, indent=4)

if __name__ == "__main__":
    main()
