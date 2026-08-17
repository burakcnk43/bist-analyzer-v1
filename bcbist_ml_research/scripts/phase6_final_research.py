import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import STOCK_UNIVERSE, DATA_FEATURES_DIR, DATA_REPORTS_DIR
from app.ml.pipeline import ResearchPipeline
from app.ml.ensemble import AlphaEnsemble
from app.research.ranking_metrics import calculate_top_k_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Phase6Final")

def main():
    logger.info("--- PHASE 6: ENSEMBLE ALPHA & HIT DISTRIBUTION ---")

    # 1. Load Data
    all_data_list = []
    for entry in STOCK_UNIVERSE:
        symbol = entry['symbol']
        path = DATA_FEATURES_DIR / f"{symbol.replace('.', '_')}_features.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, index_col='date', parse_dates=True)
            df['symbol_col'] = symbol
            all_data_list.append(df)
    global_df = pd.concat(all_data_list).sort_index()

    holdout_start = global_df.index.max() - timedelta(days=180)
    research_df = global_df[global_df.index < holdout_start]
    holdout_df = global_df[global_df.index >= holdout_start]

    h = 5
    target_cls = f'target_up_{h}d'
    target_rnk = f'target_label_{h}d'
    target_ret = f'target_return_{h}d'

    feature_patterns = ['rsi_', 'sma_', 'macd', 'bb_', 'dist_', 'atr_', 'volume', 'obv', 'rel_sector', 'sector_mean', 'macro_', 'event_']
    drop_cols = ['open', 'high', 'low', 'close', 'volume', 'adj_close', 'symbol_col', 'sector_col']
    selected_features = sorted(list(set([c for c in global_df.columns if any(p in c for p in feature_patterns)
                                       and c not in drop_cols and not c.startswith('target_')])))

    # 2. Train Ensemble Components
    logger.info("Training Multi-Task Ensemble...")

    p_cls = ResearchPipeline(model_type='xgboost', n_features=30, task='classification')
    tr_cls = research_df[selected_features + [target_cls]].dropna()
    p_cls.fit(tr_cls[selected_features], tr_cls[target_cls])

    p_rnk = ResearchPipeline(model_type='xgboost', n_features=30, task='ranking')
    tr_rnk = research_df[selected_features + [target_rnk]].dropna()
    p_rnk.fit(tr_rnk[selected_features], tr_rnk[target_rnk])

    ensemble = AlphaEnsemble(p_cls, p_rnk, w_cls=0.4, w_rnk=0.6)

    # 3. Evaluate Holdout
    test_data = holdout_df[selected_features + [target_ret]].dropna().copy()
    test_data['alpha_score'] = ensemble.get_alpha_score(test_data[selected_features])

    # 4. Generate Top-5 Metrics
    summary, daily = calculate_top_k_metrics(test_data, 'alpha_score', target_ret, k=5)

    # 5. Reporting
    report = "# Phase 6 Alpha Ranking Report\n\n"
    report += "## Top-5 Performance Summary (Holdout)\n"
    report += f"- **Average 5D Return**: {summary['avg_return']:.2%}\n"
    report += f"- **Average Excess Return**: {summary['avg_excess_return']:.2%}\n"
    report += f"- **Average Hit Rate@5**: {summary['avg_hit_rate']:.2%}\n"
    report += f"- **Win Rate (Days with positive Top-5)**: {summary['percent_days_positive']:.2%}\n\n"

    report += "## Hit Count Distribution\n"
    report += "| Hits (out of 5) | Frequency |\n"
    report += "|:---:|:---:|\n"
    for hits, freq in summary['hit_distribution'].items():
        report += f"| {int(hits)}/5 | {freq:.2%} |\n"

    path = DATA_REPORTS_DIR / "PHASE6_RANKING_RESEARCH_REPORT.md"
    with open(path, "w") as f:
        f.write(report)

    logger.info(f"Phase 6 complete. Report: {path}")
    print(report)

if __name__ == "__main__":
    main()
