import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import STOCK_UNIVERSE, DATA_FEATURES_DIR, DATA_REPORTS_DIR
from app.ml.pipeline import ResearchPipeline
from app.ml.ensemble import AlphaEnsemble
from app.research.ranking_metrics import calculate_top_k_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Phase7Research")

def main():
    logger.info("--- PHASE 7: FULL BIST UNIVERSE RANKING RESEARCH ---")

    # 1. Load All Features
    all_data_list = []
    for entry in STOCK_UNIVERSE:
        symbol = entry['symbol']
        path = DATA_FEATURES_DIR / f"{symbol.replace('.', '_')}_features.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, index_col='date', parse_dates=True)
            df['symbol_col'] = symbol
            # sector might not be in the feature file yet if built before p7
            df['sector_col'] = entry.get('sector', 'Other')
            all_data_list.append(df)
    global_df = pd.concat(all_data_list).sort_index()

    logger.info(f"Full Universe Observations: {len(global_df)}")

    # 2. Holdout & Targets
    holdout_start = global_df.index.max() - timedelta(days=180)
    research_df = global_df[global_df.index < holdout_start]
    holdout_df = global_df[global_df.index >= holdout_start]

    h = 5
    target_cls = f'target_up_{h}d'
    target_rnk = f'target_label_{h}d'
    target_ret = f'target_return_{h}d'

    feature_patterns = ['rsi_', 'sma_', 'macd', 'bb_', 'dist_', 'atr_', 'volume', 'obv', 'rel_sector', 'sector_mean', 'macro_', 'event_']
    drop_cols = ['open', 'high', 'low', 'close', 'volume', 'adj_close', 'symbol_col', 'sector_col', 'symbol', 'sector']
    selected_features = sorted(list(set([c for c in global_df.columns if any(p in c for p in feature_patterns)
                                       and c not in drop_cols and not c.startswith('target_')])))

    # --- EXPERIMENT: BIST 50 vs FULL BIST ---
    # BIST 50 subset (approximate via liquidity in holdout)
    bist50_symbols = global_df.groupby('symbol_col')['volume'].mean().nlargest(50).index.tolist()

    universes = [
        ("BIST 50 (Proxy)", bist50_symbols),
        ("FULL BIST (495)", global_df['symbol_col'].unique().tolist())
    ]

    final_leaderboard = []

    for uni_name, sym_list in universes:
        logger.info(f"Researching Universe: {uni_name}")

        # Filter data
        u_research = research_df[research_df['symbol_col'].isin(sym_list)]
        u_holdout = holdout_df[holdout_df['symbol_col'].isin(sym_list)]

        # Train Ensemble on this universe
        # LTR is better for ranking so we use w_rnk=0.7
        p_cls = ResearchPipeline(model_type='xgboost', n_features=30, task='classification')
        tr_c = u_research[selected_features + [target_cls]].dropna()
        p_cls.fit(tr_c[selected_features], tr_c[target_cls])

        p_rnk = ResearchPipeline(model_type='xgboost', n_features=30, task='ranking')
        tr_r = u_research[selected_features + [target_rnk]].dropna()
        p_rnk.fit(tr_r[selected_features], tr_r[target_rnk])

        ensemble = AlphaEnsemble(p_cls, p_rnk, w_cls=0.3, w_rnk=0.7)

        # Evaluate
        u_test = u_holdout[selected_features + [target_ret]].dropna().copy()
        u_test['alpha'] = ensemble.get_alpha_score(u_test[selected_features])

        summary, daily = calculate_top_k_metrics(u_test, 'alpha', target_ret, k=5)

        final_leaderboard.append({
            "Universe": uni_name,
            "Symbols": len(sym_list),
            "Avg_5D_Return": summary['avg_return'],
            "Excess_Ret": summary['avg_excess_return'],
            "Hit_Rate@5": summary['avg_hit_rate'],
            "Perfect_Days": summary['hit_distribution'].get(5, 0)
        })

    # 3. Report
    res_df = pd.DataFrame(final_leaderboard)
    path = DATA_REPORTS_DIR / "PHASE7_UNIVERSE_COMPARISON.csv"
    res_df.to_csv(path, index=False)

    print("\nPhase 7 Universe Comparison:")
    print(res_df.to_markdown(index=False))

    # Generate Full Markdown Report
    report = "# Phase 7: Full BIST Expansion Report\n\n"
    report += "## Universe Comparison Summary\n"
    report += res_df.to_markdown(index=False)
    report += "\n\n## Conclusion\n"

    improv = res_df.iloc[1]['Hit_Rate@5'] - res_df.iloc[0]['Hit_Rate@5']
    report += f"Expanding the universe from BIST 50 to the full eligible BIST set resulted in a **{improv:+.2%}** change in Hit Rate@5.\n"

    with open(DATA_REPORTS_DIR / "PHASE7_FULL_BIST_REPORT.md", "w") as f:
        f.write(report)

if __name__ == "__main__":
    main()
