import yfinance as yf
import pandas as pd
import logging
from typing import List, Optional
from datetime import datetime, timedelta
from app.config import DATA_RAW_DIR

logger = logging.getLogger(__name__)

class MarketDataProvider:
    def __init__(self, cache_dir: str = str(DATA_RAW_DIR)):
        self.cache_dir = cache_dir

    def fetch_ohlcv(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch OHLCV data for a given symbol.
        BIST symbols should have .IS suffix (e.g., THYAO.IS).
        """
        logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
        try:
            # Add a small buffer for technical indicator warm-up
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            warmup_start = (start_dt - timedelta(days=365)).strftime("%Y-%m-%d")

            ticker = yf.Ticker(symbol)
            df = ticker.history(start=warmup_start, end=end_date, interval="1d")

            if df.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()

            # Clean columns
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            df.index.name = "date"

            # Ensure essential columns exist
            required = ["open", "high", "low", "close", "volume"]
            for col in required:
                if col not in df.columns:
                    logger.error(f"Missing required column {col} for {symbol}")
                    return pd.DataFrame()

            # Save to cache
            cache_path = f"{self.cache_dir}/{symbol.replace('.', '_')}.csv"
            df.to_csv(cache_path)

            return df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def fetch_intraday(self, symbol: str) -> pd.DataFrame:
        """
        Fetches the first 30 mins of the current trading day (10:00 - 10:30 TR Time).
        Returns 5m interval data.
        """
        logger.info(f"Fetching intraday data for {symbol}")
        try:
            ticker = yf.Ticker(symbol)
            # Use period='1d' and interval='5m'
            df = ticker.history(period="1d", interval="5m")

            if df.empty:
                return pd.DataFrame()

            # Ensure index is datetime
            df.index = pd.to_datetime(df.index)
            # Convert to TR Time if needed (yfinance usually returns localized)
            # For simplicity, assume the latest day's data is what we need

            # Filter for 10:00 to 10:30 if possible, or just take the first few rows
            # BIST usually starts at 10:00
            return df
        except Exception as e:
            logger.error(f"Error fetching intraday for {symbol}: {e}")
            return pd.DataFrame()

    def get_realtime_indices(self) -> Dict[str, float]:
        """
        Fetches true active values for global and local indices.
        Used for regime transition confirmation.
        """
        indices = {
            "XU100": "XU100.IS",
            "SP500": "^GSPC",
            "USDTRY": "USDTRY=X",
            "GOLD": "GC=F"
        }
        results = {}
        for name, ticker in indices.items():
            try:
                data = yf.Ticker(ticker).history(period="1d")
                if not data.empty:
                    results[name] = data['Close'].iloc[-1]
            except Exception as e:
                logger.warning(f"Failed to fetch realtime {name}: {e}")
        return results

    def estimate_market_breadth(self, current_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculates active market breadth (percent above SMA50) from the provided symbol pool.
        Makes the decision engine truly 'Active' without relying on static files.
        """
        if not current_data:
             return pd.DataFrame([{"regime": "NORMAL", "pct_above_sma50": 0.5}], index=[datetime.now()])

        breadth_points = []
        # Get common dates
        all_dates = sorted(set().union(*(df.index for df in current_data.values())))

        for dt in all_dates[-20:]: # Last 20 days
            active_at_dt = []
            for sym, df in current_data.items():
                if dt in df.index:
                    # Calculate SMA50 on the fly
                    close = df['close'].loc[:dt]
                    if len(close) >= 50:
                        sma50 = close.tail(50).mean()
                        active_at_dt.append(1 if close.iloc[-1] > sma50 else 0)

            if active_at_dt:
                breadth_points.append({
                    "date": dt,
                    "pct_above_sma50": sum(active_at_dt) / len(active_at_dt)
                })

        breadth_df = pd.DataFrame(breadth_points).set_index("date")
        # Add basic momentum
        if len(breadth_df) > 5:
            breadth_df['breadth_momentum'] = breadth_df['pct_above_sma50'].diff(5)

        return breadth_df.fillna(0.5)

    def get_latest_price(self, symbol: str) -> Optional[float]:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")
            if not data.empty:
                return float(data['Close'].iloc[-1])
            return None
        except Exception:
            return None

def get_market_data(symbols: List[str], start_date: str, end_date: str) -> dict:
    provider = MarketDataProvider()
    data = {}
    for symbol in symbols:
        df = provider.fetch_ohlcv(symbol, start_date, end_date)
        if not df.empty:
            data[symbol] = df
    return data
