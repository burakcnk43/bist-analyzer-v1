import sys
import os
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import STOCK_UNIVERSE, DATA_FEATURES_DIR, DATA_REPORTS_DIR
from app.research.feature_relationships import analyze_feature_bins, analyze_stability, get_top_conditional_patterns
from app.research.regime_analysis import detect_market_regime
from app.research.event_analysis import analyze_event_impact, analyze_technical_event_interaction, analyze_event_sector_sensitivity
from app.validation.walk_forward import WalkForwardValidation
from app.research.research_report import generate_markdown_report
from app.ml.ablation import AblationTester
from app.data.event_data import get_event_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ResearchPipeline_P3")

def sanitize_for_json(data):
    if isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_json(i) for i in data]
    elif isinstance(data, (pd.Interval, pd.Timestamp)):
        return str(data)
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data) if not np.isnan(data) else None
    return data

def main():
    logger.info("Starting PHASE 3: REAL COMPANY-SPECIFIC KAP RESEARCH")

    # 1. Load Feature Datasets
    all_data_list = []
    for entry in STOCK_UNIVERSE:
        symbol = entry['symbol']
        path = DATA_FEATURES_DIR / f"{symbol.replace('.', '_')}_features.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.index.name = 'date'
            df['symbol'] = symbol
            df['sector'] = entry['sector']
            all_data_list.append(df)

    if not all_data_list:
        logger.error("No features found. Run build_features.py first.")
        return

    global_df = pd.concat(all_data_list).sort_index()

    # 2. Market Regime Detection
    logger.info("Detecting Market Regimes...")
    bist_path = os.path.join(os.path.dirname(DATA_FEATURES_DIR), 'raw', 'macro_bist100.csv')
    if os.path.exists(bist_path):
        bist_raw = pd.read_csv(bist_path, index_col='date', parse_dates=True)
        regimes = detect_market_regime(bist_raw)
        global_df['regime'] = regimes.reindex(global_df.index).ffill()
    else:
        global_df['regime'] = 'UNKNOWN'

    # 3. Ablation Testing (A to G)
    logger.info("Running Phase 3 Feature Ablation Tests...")
    ablation = AblationTester(global_df)
    ablation_results = ablation.run_tests()
    ablation_results.to_csv(DATA_REPORTS_DIR / "kap_model_ablation.csv", index=False)

    # 4. Company-Specific Event Analysis
    logger.info("Analyzing Company-Specific Event Impact...")
    symbols = [s['symbol'] for s in STOCK_UNIVERSE]
    all_dates = global_df.index.unique()
    events_df = get_event_dataset(symbols, all_dates.min().strftime("%Y-%m-%d"), all_dates.max().strftime("%Y-%m-%d"))

    event_type_stats = analyze_event_impact(global_df, events_df)
    event_type_stats.to_csv(DATA_REPORTS_DIR / "event_type_analysis.csv", index=False)

    sector_sensitivity = analyze_event_sector_sensitivity(global_df)
    sector_sensitivity.to_csv(DATA_REPORTS_DIR / "sector_event_analysis.csv", index=False)

    tech_interaction = analyze_technical_event_interaction(global_df)

    # 5. Data Quality Report
    logger.info("Generating KAP Data Quality Audit...")
    quality_data = []
    for symbol in global_df['symbol'].unique():
        sdf = global_df[global_df['symbol'] == symbol]
        quality_data.append({
            "symbol": symbol,
            "total_events_20d_avg": sdf['event_count_20d'].mean(),
            "filling_rate": (sdf['event_count_20d'] > 0).mean()
        })
    quality_df = pd.DataFrame(quality_data)
    quality_df.to_csv(DATA_REPORTS_DIR / "kap_feature_quality.csv", index=False)

    # 6. Construct Final Report Data
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "stocks_analyzed": len(STOCK_UNIVERSE),
        "total_samples": len(global_df),
        "real_kap_events": len(events_df[events_df['symbol'] != "BIST100"]) if not events_df.empty else 0,
        "ablation_results": ablation_results.to_dict('records'),
        "event_type_analysis": event_type_stats.head(20).to_dict('records') if not event_type_stats.empty else [],
        "sector_sensitivity": sector_sensitivity.to_dict('records') if not sector_sensitivity.empty else [],
        "tech_event_interaction": tech_interaction.to_dict('records') if not tech_interaction.empty else []
    }

    report_data = sanitize_for_json(report_data)

    # Save JSON
    with open(DATA_REPORTS_DIR / "PHASE3_COMPANY_KAP_REPORT.json", 'w') as f:
        json.dump(report_data, f, indent=4)

    # 7. Markdown Generation
    logger.info("Generating Phase 3 Markdown Report...")
    report_md = f"""
# PHASE 3 RESEARCH REPORT - REAL COMPANY-SPECIFIC KAP ANALYSIS
**Generated on:** {report_data['timestamp']}

## 1. Executive Summary
- **Total Real Company Events:** {report_data['real_kap_events']}
- **KAP Status:** {"OPERATIONAL (CSV/Real)" if report_data['real_kap_events'] > 0 else "LIMITED (Macro Only)"}

## 2. Model Ablation (Alpha Contribution)
| Step | Configuration | Accuracy | Gain |
|------|---------------|----------|------|
"""
    prev_acc = 0
    for res in report_data['ablation_results']:
        gain = res['Avg Accuracy'] - prev_acc if prev_acc > 0 else 0
        report_md += f"| {res['Configuration'][:2]} | {res['Configuration']} | {res['Avg Accuracy']:.4f} | {gain:+.4f} |\n"
        prev_acc = res['Avg Accuracy']

    report_md += """
## 3. High-Impact Event Types
Types associated with the strongest 5-day post-event moves.
| Event Type | Samples | Mean Return | Win Rate |
|------------|---------|-------------|----------|
"""
    for ev in report_data['event_type_analysis']:
        report_md += f"| {ev['event_type']} | {ev['sample_count']} | {ev['mean_return']*100:.2f}% | {ev['win_rate']*100:.2f}% |\n"

    report_md += """
## 4. Sector Sensitivity Ranking
Sectors responding most strongly to corporate disclosures.
| Sector | Samples | Mean Return | Win Rate |
|--------|---------|-------------|----------|
"""
    for sec in report_data['sector_sensitivity']:
        report_md += f"| {sec['sector']} | {sec['count']} | {sec['mean_return']*100:.2f}% | {sec['win_rate']*100:.2f}% |\n"

    report_md += """
## 5. Technical + KAP Interaction
| Technical State | Has Event | Mean Return | Win Rate |
|-----------------|-----------|-------------|----------|
"""
    for inter in report_data['tech_event_interaction']:
        report_md += f"| {inter['rsi_state']} | {inter['has_event']} | {inter['mean_return']*100:.2f}% | {inter['win_rate']*100:.2f}% |\n"

    report_md += """
## 6. Conclusions
- **Alpha Decay:** We observe whether KAP impact is immediate (1D) or sustained (20D).
- **Synergy:** Technical indicators like RSI oversold show different win rates when combined with news.
"""

    with open(DATA_REPORTS_DIR / "PHASE3_COMPANY_KAP_REPORT.md", 'w') as f:
        f.write(report_md)

    logger.info(f"Phase 3 complete. Report: {DATA_REPORTS_DIR / 'PHASE3_COMPANY_KAP_REPORT.md'}")

if __name__ == "__main__":
    main()
