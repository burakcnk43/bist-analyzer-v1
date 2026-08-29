import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ml.pdf_reports import BcbistPdfEngine

def test_pdf():
    engine = BcbistPdfEngine(Path("data/reports/daily"))
    sample_data = {
        "date": "2026-08-29",
        "market_regime": "BULL",
        "market_quality": 85.5,
        "drift_score": 0.02,
        "selected_k": 3,
        "status": "SUCCESS",
        "predictions": [
            {"rank": 1, "symbol": "THYAO.IS", "sector": "Aviation", "probability": 0.88, "reason": "Strong momentum confirmation"},
            {"rank": 2, "symbol": "AKBNK.IS", "sector": "Banking", "probability": 0.82, "reason": "Relative strength leader"},
            {"rank": 3, "symbol": "EREGL.IS", "sector": "Steel", "probability": 0.75, "reason": "Above 20-day mean"}
        ]
    }

    output_path = engine.generate_daily_pdf(sample_data)
    print(f"PDF Generated at: {output_path}")
    if output_path.exists():
        print("Verification: SUCCESS")
    else:
        print("Verification: FAILED")

if __name__ == "__main__":
    test_pdf()
