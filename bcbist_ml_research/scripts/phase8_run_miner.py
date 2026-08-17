import sys
import os
import pandas as pd
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import DATA_REPORTS_DIR
from app.research.pattern_miner import PatternMiner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PatternMining")

def main():
    log_path = DATA_REPORTS_DIR / "PHASE8_PREDICTION_LOG.csv"
    if not log_path.exists():
        logger.error("Prediction log not found. Run scripts/phase8_generate_prediction_log.py first.")
        return

    miner = PatternMiner(str(log_path))

    logger.info("Mining success patterns via Decision Tree...")
    rules = miner.mine_success_patterns(min_samples=100)

    logger.info("Analyzing sector robustness...")
    sector_stats = miner.analyze_sector_performance()

    logger.info("Analyzing RSI success ranges...")
    rsi_stats = miner.analyze_feature_ranges('rsi_14', bins=5)

    logger.info("Analyzing Volume success ranges...")
    vol_stats = miner.analyze_feature_ranges('relative_volume', bins=5)

    # 4. Generate Report
    report = "# Phase 8: Success Pattern Mining Report\n\n"

    report += "## 1. Discovered Decision Rules (Top-10 Picks)\n"
    report += "These rules describe conditions where Top-10 model picks were more likely to be Hits (>0% return).\n"
    report += "```\n" + rules + "\n```\n\n"

    report += "## 2. Sector-Specific Hit Rates (Top-5 Only)\n"
    report += sector_stats.to_markdown() + "\n\n"

    report += "## 3. Indicator Optimization (Quantile Analysis)\n"
    report += "### RSI 14 Performance Buckets\n"
    report += rsi_stats.to_markdown() + "\n\n"

    report += "### Relative Volume Performance Buckets\n"
    report += vol_stats.to_markdown() + "\n\n"

    report_path = DATA_REPORTS_DIR / "PHASE8_SUCCESS_PATTERNS.md"
    with open(report_path, "w") as f:
        f.write(report)

    logger.info(f"Success patterns report generated at {report_path}")

if __name__ == "__main__":
    main()
