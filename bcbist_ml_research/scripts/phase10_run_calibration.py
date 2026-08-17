import sys
import os
import pandas as pd
import numpy as np
import logging
import joblib

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import DATA_REPORTS_DIR, MODELS_DIR
from app.ml.calibration import ProbabilityCalibrator, calculate_calibration_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CalibrationRunner")

def main():
    log_path = DATA_REPORTS_DIR / "PHASE10_MULTI_HORIZON_LOG.csv"
    if not log_path.exists():
        logger.error("Multi-horizon log not found.")
        return

    df = pd.read_csv(log_path)
    horizons = [1, 3, 5]

    report = "# Phase 10: Probability Calibration Report (Isotonic)\n\n"

    for h in horizons:
        logger.info(f"Calibrating Horizon {h}d...")
        score_col = f'score_{h}d'
        actual_col = f'actual_{h}d'

        if score_col not in df.columns: continue

        y = (df[actual_col] > 0).astype(int).values
        X = df[score_col].values

        # Use Isotonic Regression to preserve rank perfectly
        calibrator = ProbabilityCalibrator(method='isotonic')
        calibrator.fit(X, y)

        probs = calibrator.calibrate(X)
        metrics = calculate_calibration_metrics(y, probs)

        joblib.dump(calibrator, MODELS_DIR / f"phase10_calibrator_{h}d.joblib")

        report += f"## Horizon {h}d\n"
        report += f"- **Expected Calibration Error (ECE)**: {metrics['ece']:.4f}\n\n"

        bins = pd.cut(probs, bins=10, duplicates='drop')
        bin_df = pd.DataFrame({'pred': probs, 'actual': y, 'bin': bins})
        stats = bin_df.groupby('bin', observed=False).agg({'actual': ['mean', 'count'], 'pred': 'mean'})
        stats.columns = ['Actual_Prob', 'Sample_Count', 'Avg_Pred_Prob']
        report += stats.to_markdown() + "\n\n"

    with open(DATA_REPORTS_DIR / "PHASE10_CALIBRATION_REPORT.md", "w") as f:
        f.write(report)
    logger.info(f"Calibration report saved to {DATA_REPORTS_DIR}")

if __name__ == "__main__":
    main()
