import pandas as pd
import logging
import requests
import time
from datetime import datetime
from typing import List, Optional
from app.config import DATA_RAW_DIR

logger = logging.getLogger(__name__)

# Standard event types
EVENT_TYPES = [
    "financial_results", "dividend", "capital_increase", "share_buyback",
    "contract", "tender", "investment", "acquisition", "partnership",
    "credit_rating", "management_change", "legal", "operational",
    "guidance", "central_bank", "geopolitical", "other"
]

class Event:
    def __init__(self, symbol: str, timestamp: datetime, event_type: str, title: str,
                 importance: int = 1, category: str = "other", description: str = "",
                 sentiment: float = 0.0, source: str = "KAP"):
        self.symbol = symbol
        self.timestamp = timestamp
        self.date = timestamp.date()
        self.event_type = event_type if event_type in EVENT_TYPES else "other"
        self.category = category
        self.title = title
        self.description = description
        self.importance = importance # 1 to 5
        self.sentiment = sentiment # -1.0 to 1.0
        self.source = source

class BaseEventProvider:
    def fetch_events(self, symbols: List[str], start_date: str, end_date: str) -> List[Event]:
        raise NotImplementedError

class MacroEventProvider(BaseEventProvider):
    def fetch_events(self, symbols: List[str], start_date: str, end_date: str) -> List[Event]:
        events = []
        cbrt_dates = [
            "2023-01-19", "2023-02-23", "2023-03-23", "2023-04-27", "2023-05-25",
            "2023-06-22", "2023-07-20", "2023-08-24", "2023-09-21", "2023-10-26",
            "2023-11-23", "2023-12-21", "2024-01-25", "2024-02-22", "2024-03-21",
            "2024-04-25", "2024-05-23", "2024-06-27", "2024-07-23"
        ]
        for d in cbrt_dates:
            ts = pd.to_datetime(d + " 14:00:00")
            if ts >= pd.to_datetime(start_date) and ts <= pd.to_datetime(end_date):
                events.append(Event(
                    symbol="BIST100",
                    timestamp=ts,
                    event_type="central_bank",
                    category="interest_rate",
                    title="CBRT Interest Rate Decision",
                    importance=5,
                    source="CBRT"
                ))
        return events

class KAPProvider(BaseEventProvider):
    def __init__(self):
        self.api_url = "https://www.kap.org.tr/tr/api/disclosures"
        self.timeout_conn = 10
        self.timeout_read = 20
        self.max_retries = 2

    def fetch_events(self, symbols: List[str], start_date: str, end_date: str) -> List[Event]:
        """
        Attempts to fetch recent disclosures from KAP JSON API with strict timeouts.
        """
        logger.info(f"KAPProvider: Attempting to fetch recent disclosures...")

        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()
                response = requests.get(
                    self.api_url,
                    timeout=(self.timeout_conn, self.timeout_read)
                )
                duration = time.time() - start_time

                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"KAPProvider: Success. Retrieved {len(data)} items in {duration:.2f}s.")
                    # In a real implementation, we would filter by symbol and parse fields.
                    # For research integrity, if we can't parse reliable historical dates for 50 stocks here,
                    # we only use this for 'live' or 'very recent' updates.
                    return [] # Placeholder: parsing logic for KAP's specific JSON schema goes here
                else:
                    logger.warning(f"KAPProvider: Status {response.status_code} on attempt {attempt+1}")
            except requests.exceptions.RequestException as e:
                logger.error(f"KAPProvider: Connection error on attempt {attempt+1}: {e}")

            if attempt < self.max_retries:
                time.sleep(2 ** attempt) # Exponential backoff

        return []

class CSVFallbackProvider(BaseEventProvider):
    def __init__(self, file_path=DATA_RAW_DIR / "kap_events.csv"):
        self.file_path = file_path

    def fetch_events(self, symbols: List[str], start_date: str, end_date: str) -> List[Event]:
        if not self.file_path.exists():
            logger.warning(f"CSVFallbackProvider: {self.file_path} not found.")
            return []

        try:
            df = pd.read_csv(self.file_path)
            events = []
            for _, row in df.iterrows():
                if row['symbol'] in symbols:
                    ts = pd.to_datetime(row['timestamp'])
                    if ts >= pd.to_datetime(start_date) and ts <= pd.to_datetime(end_date):
                        events.append(Event(
                            symbol=row['symbol'],
                            timestamp=ts,
                            event_type=row.get('event_type', 'other'),
                            title=row.get('title', 'KAP Disclosure'),
                            category=row.get('category', 'other'),
                            importance=int(row.get('importance', 1)),
                            sentiment=float(row.get('sentiment', 0.0)),
                            source="Local CSV"
                        ))
            return events
        except Exception as e:
            logger.error(f"CSVFallbackProvider: Error reading file: {e}")
            return []

from app.data.news_data import NewsDataProvider

class NewsEventProvider(BaseEventProvider):
    def __init__(self):
        self.provider = NewsDataProvider()

    def fetch_events(self, symbols: List[str], start_date: str, end_date: str) -> List[Event]:
        all_news = []
        for sym in symbols[:30]: # Limit for performance
            news_items = self.provider.fetch_news(sym)
            for n in news_items:
                ts = pd.to_datetime(n['timestamp'])
                if ts >= pd.to_datetime(start_date) and ts <= pd.to_datetime(end_date):
                    all_news.append(Event(
                        symbol=n['symbol'],
                        timestamp=ts,
                        event_type="news",
                        title=n['title'],
                        sentiment=n['sentiment'],
                        source=n['source']
                    ))
        return all_news

class EventDataProvider:
    def __init__(self):
        self.providers = [
            MacroEventProvider(),
            KAPProvider(),
            NewsEventProvider(),
            CSVFallbackProvider()
        ]

    def fetch_all(self, symbols: List[str], start_date: str, end_date: str) -> List[Event]:
        all_events = []
        for provider in self.providers:
            try:
                events = provider.fetch_events(symbols, start_date, end_date)
                all_events.extend(events)
            except Exception as e:
                logger.error(f"Error in provider {provider.__class__.__name__}: {e}")
        return all_events

def get_event_dataset(symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    orchestrator = EventDataProvider()
    events = orchestrator.fetch_all(symbols, start_date, end_date)

    if not events:
        return pd.DataFrame()

    data = []
    for e in events:
        data.append({
            "symbol": e.symbol,
            "timestamp": e.timestamp,
            "date": e.date,
            "event_type": e.event_type,
            "category": e.category,
            "importance": e.importance,
            "sentiment": e.sentiment,
            "title": e.title,
            "source": e.source
        })

    return pd.DataFrame(data).sort_values("timestamp")
