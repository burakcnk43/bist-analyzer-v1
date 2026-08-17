import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def check_data_quality(df: pd.DataFrame, symbol: str) -> dict:
    """
    Performs data quality checks on a stock dataframe.
    """
    report = {
        "symbol": symbol,
        "row_count": len(df),
        "missing_values": df.isnull().sum().to_dict(),
        "zero_volume_days": (df['volume'] == 0).sum() if 'volume' in df.columns else 0,
        "price_anomalies": 0,
        "status": "PASS"
    }

    # Check for negative prices
    if 'close' in df.columns and (df['close'] <= 0).any():
        report["price_anomalies"] += (df['close'] <= 0).sum()
        report["status"] = "FAIL"

    # Check for extreme outliers (daily return > 50% for BIST is rare outside corporate actions)
    if 'close' in df.columns:
        returns = df['close'].pct_change()
        outliers = (returns.abs() > 0.5).sum()
        report["price_anomalies"] += outliers
        if outliers > 0:
            report["status"] = "WARNING"

    return report
