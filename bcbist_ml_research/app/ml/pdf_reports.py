from fpdf import FPDF
import pandas as pd
from pathlib import Path
from datetime import datetime

class BcbistPdfEngine:
    """
    Generates professional PDF intelligence reports for daily stock picks.
    """
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_daily_pdf(self, data: dict) -> Path:
        date_str = data['date'][:10]
        pdf = FPDF()
        pdf.add_page()

        # Header
        pdf.set_font("Helvetica", "B", 24)
        pdf.set_text_color(20, 50, 100)
        pdf.cell(0, 20, "BCBIST Ultra-Short Intelligence", ln=True, align="C")

        pdf.set_font("Helvetica", "", 12)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 10, f"Date: {date_str} | Horizon: 24-48 Hours", ln=True, align="C")
        pdf.ln(10)

        # Market Context
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, "1. Market Environment", ln=True)

        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 8, f"Regime: {data['market_regime']}", ln=True)
        pdf.cell(0, 8, f"Quality Score: {data['market_quality']:.2f}/100", ln=True)
        pdf.cell(0, 8, f"Drift Index: {data['drift_score']:.4f}", ln=True)
        pdf.ln(10)

        # Top Selections
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, f"2. Top {data['selected_k']} Selections", ln=True)

        # Table Header
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(20, 10, "Rank", 1, 0, "C", fill=True)
        pdf.cell(40, 10, "Symbol", 1, 0, "C", fill=True)
        pdf.cell(40, 10, "Sector", 1, 0, "C", fill=True)
        pdf.cell(30, 10, "Prob", 1, 0, "C", fill=True)
        pdf.cell(60, 10, "Key Driver", 1, 1, "C", fill=True)

        # Table Rows
        pdf.set_font("Helvetica", "", 10)
        for p in data['predictions']:
            pdf.cell(20, 10, str(p['rank']), 1, 0, "C")
            pdf.cell(40, 10, p['symbol'], 1, 0, "C")
            pdf.cell(40, 10, p['sector'], 1, 0, "C")
            pdf.cell(30, 10, f"{p['probability']:.2%}", 1, 0, "C")
            # Truncate reason if too long
            reason = p['reason'][:35] + "..." if len(p['reason']) > 35 else p['reason']
            pdf.cell(60, 10, reason, 1, 1, "L")

        pdf.ln(10)

        # Risk Note
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(150, 0, 0)
        pdf.multi_cell(0, 5, "Disclaimer: These predictions are based on statistical modeling and do not constitute financial advice. Always verify with current market depth.")

        output_path = self.output_dir / f"BCBIST_REPORT_{date_str}.pdf"
        pdf.output(str(output_path))

        return output_path
