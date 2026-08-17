import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import joblib

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import STOCK_UNIVERSE, DATA_FEATURES_DIR, DATA_REPORTS_DIR, MODELS_DIR
from app.ml.production_scorer import ProductionScorer
from app.research.ranking_metrics import calculate_top_k_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FreshValidation")

def main():
    logger.info("--- PHASE 10: FRESH FORWARD VALIDATION (AUG 5 - AUG 13) ---")

    # 1. Load Data
    all_data_list = []
    for entry in STOCK_UNIVERSE:
        symbol = entry['symbol']
        path = DATA_FEATURES_DIR / f"{symbol.replace('.', '_')}_features.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, index_col='date', parse_dates=True)
            df['symbol_col'] = symbol
            df['sector_col'] = entry.get('sector', 'Other')
            all_data_list.append(df)

    if not all_data_list:
        logger.error("No feature data found.")
        return

    global_df = pd.concat(all_data_list).sort_index()

    # 2. Define Fresh Period
    fresh_start = '2026-08-05'
    fresh_df = global_df[global_df.index >= fresh_start].copy()

    if fresh_df.empty:
        logger.error("No data found for fresh period starting 2026-08-05.")
        return

    # 3. Load Frozen Assets
    logger.info("Loading Frozen Phase 10 Assets...")
    p10_scorer = ProductionScorer(MODELS_DIR)
    success_mdl = joblib.load(MODELS_DIR / "phase10_success_predictor.joblib")
    sim_engine = joblib.load(MODELS_DIR / "phase10_similarity_engine.joblib")

    # Load Phase 9 for side-by-side
    p_cls_9 = joblib.load(MODELS_DIR / "phase10_cls_5d.joblib")
    p_rnk_9 = joblib.load(MODELS_DIR / "phase10_rnk_5d.joblib")
    from app.ml.ensemble import AlphaEnsemble
    ens_9 = AlphaEnsemble(p_cls_9, p_rnk_9, w_cls=0.3, w_rnk=0.7)

    # 4. Simulation Loop
    dates = fresh_df.index.unique().sort_values()
    logger.info(f"Simulating {len(dates)} fresh trading days...")

    feature_patterns = ['rsi_', 'sma_', 'macd', 'bb_', 'dist_', 'atr_', 'volume', 'obv', 'rel_sector', 'sector_mean', 'macro_', 'event_']
    selected_features = sorted(list(set([c for c in global_df.columns if any(p in c for p in feature_patterns)
                                       and not any(x in c for x in ['target', 'symbol', 'sector', 'col'])])))

    fresh_predictions = []

    for date in dates:
        day_data = fresh_df.loc[[date]]
        mkt_ctx = {'market_z_rsi_14': day_data['market_z_rsi_14'].iloc[0] if 'market_z_rsi_14' in day_data.columns else 0}

        # Evidence signals
        sim_scores = sim_engine.get_similarity_feature(day_data[selected_features])

        # P10 Selection
        scores_10 = p10_scorer.calculate_production_scores(
            day_data[selected_features],
            mkt_ctx,
            success_mdl=success_mdl,
            sim_alpha=sim_scores
        )
        top5_10 = p10_scorer.select_top_5(scores_10, day_data[['symbol_col', 'sector_col']])

        # P9 Selection (Baseline)
        scores_9 = ens_9.get_alpha_score(day_data[selected_features])
        day_9 = day_data.copy()
        day_9['p9_score'] = scores_9
        top5_9 = day_9.sort_values('p9_score', ascending=False).head(5)

        # Log Predictions
        for rank, (idx, row) in enumerate(top5_10.iterrows()):
            fresh_predictions.append({
                "date": date,
                "model": "PHASE10",
                "symbol": row['symbol_col'],
                "rank": rank + 1,
                "prob_5d": row.get('prob_5d', 0),
                "alpha": row.get('production_alpha', 0),
                "regime": row.get('regime', 'NORMAL')
            })

        for rank, (idx, row) in enumerate(top5_9.iterrows()):
            fresh_predictions.append({
                "date": date,
                "model": "PHASE9",
                "symbol": row['symbol_col'],
                "rank": rank + 1,
                "alpha": row['p9_score']
            })

    pred_df = pd.DataFrame(fresh_predictions)
    pred_df.to_csv(DATA_REPORTS_DIR / "PHASE10_FRESH_PREDICTIONS.csv", index=False)

    # 5. Outcome Calculation
    logger.info("Calculating outcomes...")
    results_list = []

    for date in dates:
        # P10 Outcomes
        p10_day = pred_df[(pred_df['date'] == date) & (pred_df['model'] == "PHASE10")]
        p9_day = pred_df[(pred_df['date'] == date) & (pred_df['model'] == "PHASE9")]

        day_metrics = {"date": date}

        for name, d_picks in [("P10", p10_day), ("P9", p9_day)]:
            if d_picks.empty:
                day_metrics[f"{name}_hits"] = 0
                day_metrics[f"{name}_ret"] = 0
                continue

            # Match with global_df to get actuals
            symbols = d_picks['symbol'].tolist()
            actuals = global_df.loc[date].set_index('symbol_col').loc[symbols]

            # Hit defined as target_return_5d > 0
            hits = (actuals['target_return_5d'] > 0).sum()
            avg_ret = actuals['target_return_5d'].mean()

            day_metrics[f"{name}_hits"] = hits
            day_metrics[f"{name}_ret"] = avg_ret
            day_metrics[f"{name}_3PLUS"] = 1 if hits >= 3 else 0

        results_list.append(day_metrics)

    res_df = pd.DataFrame(results_list)
    res_df.to_csv(DATA_REPORTS_DIR / "PHASE10_FRESH_RESULTS.csv", index=False)

    # 6. Audit & Scoreboard
    print("\n" + "="*60)
    print("BCBIST PHASE 10 FINAL REALITY CHECK")
    print(f"Period: {dates.min().date()} to {dates.max().date()}")
    print("="*60)

    valid_5d_days = res_df.dropna(subset=['P10_ret'])
    if len(valid_5d_days) < 20:
        print("VERDICT: INSUFFICIENT_FORWARD_SAMPLE")

    print("\nSCOREBOARD:")
    print(f"PHASE 9  - Avg 5D Hit Rate: {res_df['P9_hits'].mean()/5:.2%}")
    print(f"PHASE 10 - Avg 5D Hit Rate: {res_df['P10_hits'].mean()/5:.2%}")
    print(f"PHASE 9  - Top5_3PLUS:     {res_df['P9_3PLUS'].mean():.2%}")
    print(f"PHASE 10 - Top5_3PLUS:     {res_df['P10_3PLUS'].mean():.2%}")
    print(f"PHASE 10 - Avg 5D Return:  {res_df['P10_ret'].mean():.2%}")
    print("="*60)

    # 7. Final Report Construction
    report = f"# Phase 10: Fresh Forward Validation Report\n\n"
    report += f"**Period**: {dates.min().date()} to {dates.max().date()}\n"
    report += f"**Trading Days**: {len(dates)}\n\n"
    report += "## Scoreboard\n"
    report += "| Metric | Phase 9 | Phase 10 | Delta |\n"
    report += "|:---|:---:|:---:|:---:|\n"
    report += f"| Hit Rate@5 | {res_df['P9_hits'].mean()/5:.2%} | {res_df['P10_hits'].mean()/5:.2%} | {(res_df['P10_hits'].mean() - res_df['P9_hits'].mean())/5:+.2%} |\n"
    report += f"| Top5_3PLUS | {res_df['P9_3PLUS'].mean():.2%} | {res_df['P10_3PLUS'].mean():.2%} | {res_df['P10_3PLUS'].mean() - res_df['P9_3PLUS'].mean():+.2%} |\n"
    report += f"| Avg Return | {res_df['P9_ret'].mean():.2%} | {res_df['P10_ret'].mean():.2%} | {res_df['P10_ret'].mean() - res_df['P9_ret'].mean():+.2%} |\n\n"

    verdict = "PHASE10_PROMISING_BUT_INSUFFICIENT_SAMPLE" if len(dates) < 20 else "PHASE10_VALIDATED"
    report += f"## Final Verdict\n**{verdict}**\n"

    with open(DATA_REPORTS_DIR / "PHASE10_FRESH_FORWARD_REPORT.md", "w") as f:
        f.write(report)

if __name__ == "__main__":
    main()
