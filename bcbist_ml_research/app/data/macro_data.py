import yfinance as yf
import pandas as pd
import logging
from typing import Dict
from app.config import DATA_RAW_DIR

logger = logging.getLogger(__name__)

class MacroDataProvider:
    MACRO_SYMBOLS = {
        "USDTRY": "USDTRY=X",
        "EURTRY": "EURTRY=X",
        "BIST100": "XU100.IS",
        "GOLD": "GC=F",
        "BRENT": "BZ=F",
        "SP500": "^GSPC"
    }

    def fetch_macro_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        data = {}
        for key, symbol in self.MACRO_SYMBOLS.items():
            try:
                logger.info(f"Fetching macro data for {key} ({symbol})")
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)
                if not df.empty:
                    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
                    df.index.name = "date"
                    data[key] = df
            except Exception as e:
                logger.error(f"Error fetching macro {key}: {str(e)}")
        return data

def get_macro_features(start_date: str, end_date: str) -> pd.DataFrame:
    provider = MacroDataProvider()
    macros = provider.fetch_macro_data(start_date, end_date)

    combined = pd.DataFrame()
    for key, df in macros.items():
        if not df.empty:
            # We mostly care about the close price and return
            series = df['close'].rename(f"macro_{key.lower()}_close")
            ret_series = df['close'].pct_change().rename(f"macro_{key.lower()}_return")

            if combined.empty:
                combined = pd.concat([series, ret_series], axis=1)
            else:
                combined = combined.join(pd.concat([series, ret_series], axis=1), how='outer')

    return combined.ffill()
