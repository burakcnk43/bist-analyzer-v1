import os
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Database & Storage
FILES_DIR = Path(os.getenv("FILES_DIR", "."))
DATA_RAW_DIR = Path(os.getenv("DATA_RAW_DIR", str(BASE_DIR / "data" / "raw")))
DATA_PROCESSED_DIR = FILES_DIR / "data" / "processed"
DATA_FEATURES_DIR = FILES_DIR / "data" / "features"
DATA_LABELS_DIR = FILES_DIR / "data" / "labels"
DATA_REPORTS_DIR = Path(os.getenv("DATA_REPORTS_DIR", str(BASE_DIR / "data" / "reports")))
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(BASE_DIR / "models")))

# Create directories if they don't exist
for d in [DATA_PROCESSED_DIR, DATA_FEATURES_DIR, DATA_LABELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Universe Loading
VALID_PATH = DATA_RAW_DIR / "bist_universe_valid.csv"
UNIVERSE_PATH = DATA_RAW_DIR / "bist_universe.csv"

def load_universe():
    if VALID_PATH.exists():
        df = pd.read_csv(VALID_PATH)
        return df.to_dict('records')
    elif UNIVERSE_PATH.exists():
        df = pd.read_csv(UNIVERSE_PATH)
        return df.to_dict('records')
    else:
        return [{"symbol": "THYAO.IS", "sector": "Aviation"}]

STOCK_UNIVERSE = load_universe()

# Production Config
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
PREDICTION_HORIZONS = [1, 3, 5, 10, 20]
TRAIN_TEST_SPLIT_DATE = os.getenv("TRAIN_TEST_SPLIT_DATE", "2024-01-01")

# API Configuration
PORT = int(os.getenv("PORT", 8000))
MARKET_DATA_PROVIDER = os.getenv("MARKET_DATA_PROVIDER", "yfinance")
BREADTH_DATA_PATH = Path(__file__).parent.parent / "market_breadth.csv"
