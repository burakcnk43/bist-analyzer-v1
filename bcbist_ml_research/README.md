# BCBIST ML Research Engine

A senior-level machine learning research engine for Borsa Istanbul (BIST). This platform is designed to discover deep, non-linear relationships between financial variables and stock market behavior while maintaining strict integrity through leakage-safe validation.

## Core Features
- **Leakage-Safe Architecture**: Uses walk-forward validation with purging and embargoing.
- **Multi-Source Data**: Integrates market data (yfinance), fundamentals, news sentiment, macro indicators, and geopolitical events.
- **Deep Feature Engineering**: 100+ features including technical indicators, volume dynamics, sector relative strength, and regime-based features.
- **Explainable AI**: Sector-specific model importance, SHAP analysis, and quantile-based return distributions.
- **Deployment Ready**: Fully containerized for Railway with FastAPI backend.

## Project Structure
- `app/`: Core application logic (API, Data, Features, ML, Research).
- `scripts/`: Operational scripts for data collection, training, and research.
- `data/`: Storage for raw, processed, and feature-engineered datasets.
- `models/`: Saved ML models and metadata.

## Setup
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Configure `.env` (use `.env.example` as a template).
4. Run research: `python scripts/run_research.py`
5. Start API: `uvicorn app.main:app --reload`

## Methodology
The engine does not use hardcoded rules (e.g., RSI > 70). Instead, it learns from historical patterns:
1. **Purged Walk-Forward**: Ensures that testing is always done on out-of-sample data with gaps to prevent overlap leakage.
2. **Sector Context**: Models are trained both globally and per-sector to capture industry-specific nuances.
3. **Regime Awareness**: Features are analyzed under different market conditions (Bull/Bear/Volatile).
