from fastapi import APIRouter, Request, HTTPException
import pandas as pd
import numpy as np
import logging
from app.config import DATA_FEATURES_DIR, BREADTH_DATA_PATH

from fastapi.responses import FileResponse
from app.data.market_data import MarketDataProvider

router = APIRouter()
logger = logging.getLogger(__name__)
market_provider = MarketDataProvider()

def discover_latest_date():
    """
    Finds the latest common date across feature files.
    """
    try:
        if BREADTH_DATA_PATH.exists():
            mkt_df = pd.read_csv(BREADTH_DATA_PATH, index_col='date', parse_dates=True)
            if not mkt_df.empty:
                return mkt_df.index.max().strftime("%Y-%m-%d")

        latest_dates = []
        for f in list(DATA_FEATURES_DIR.glob("*.csv"))[:10]:
            with open(f, 'r') as file:
                 lines = file.readlines()
                 if len(lines) > 1:
                     last_line = lines[-1]
                     latest_dates.append(last_line.split(',')[0])

        if not latest_dates:
            return None
        return max(latest_dates)
    except Exception as e:
        logger.error(f"Error discovering latest date: {e}")
        return None

@router.get("/daily-picks")
async def get_daily_picks(request: Request, intraday: bool = False):
    pm = request.app.state.production_manager

    try:
        latest_date = discover_latest_date()
        if not latest_date:
            raise HTTPException(status_code=503, detail="Decision engine not ready")

        all_dfs = []
        symbols_found = []
        count = 0
        for f in DATA_FEATURES_DIR.glob("*.csv"):
            if count > 150: break
            df = pd.read_csv(f, index_col='date', parse_dates=True)
            if latest_date in df.index:
                sym = f.stem.replace('_features', '').replace('_', '.')
                row = df.loc[[latest_date]].copy()
                row['symbol_col'] = sym
                all_dfs.append(row)
                symbols_found.append(sym)
                count += 1

        if not all_dfs:
            raise HTTPException(status_code=404, detail="No symbols found")

        day_data = pd.concat(all_dfs)
        mkt_df = pd.read_csv(BREADTH_DATA_PATH, index_col='date', parse_dates=True)
        mkt_data = mkt_df[mkt_df.index <= latest_date].tail(20)

        # Fetch Intraday if requested
        intraday_results = None
        if intraday:
            logger.info("Fetching intraday confirmation data...")
            intraday_results = {}
            for sym in symbols_found[:50]:
                intra = market_provider.fetch_intraday(sym)
                if not intra.empty:
                    intraday_results[sym] = intra

        response = pm.get_daily_picks(day_data, mkt_data, intraday_data=intraday_results)
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal decision engine error")

@router.get("/report-pdf")
async def get_latest_report_pdf(request: Request):
    """
    Downloads the latest generated PDF report.
    """
    report_dir = Path("data/reports/daily")
    pdfs = sorted(report_dir.glob("*.pdf"))
    if not pdfs:
        raise HTTPException(status_code=404, detail="No reports generated yet")

    latest_pdf = pdfs[-1]
    return FileResponse(latest_pdf, media_type='application/pdf', filename=latest_pdf.name)

@router.get("/{symbol}")
async def get_prediction(symbol: str, request: Request):
    pm = request.app.state.production_manager
    symbol_raw = symbol.upper()
    symbol_is = symbol_raw if symbol_raw.endswith(".IS") else f"{symbol_raw}.IS"

    try:
        latest_date = discover_latest_date()
        feat_path = DATA_FEATURES_DIR / f"{symbol_is.replace('.', '_')}_features.csv"

        if not feat_path.exists():
             raise HTTPException(status_code=404, detail=f"Symbol {symbol_raw} not in eligible universe")

        df = pd.read_csv(feat_path, index_col='date', parse_dates=True)
        if latest_date not in df.index:
             raise HTTPException(status_code=404, detail=f"Data for {symbol_raw} at {latest_date} not available")

        # Full context evaluation
        all_dfs = []
        count = 0
        for f in DATA_FEATURES_DIR.glob("*.csv"):
            if count > 100: break
            sub_df = pd.read_csv(f, index_col='date', parse_dates=True)
            if latest_date in sub_df.index:
                row = sub_df.loc[[latest_date]].copy()
                row['symbol_col'] = f.stem.replace('_features', '').replace('_', '.')
                all_dfs.append(row)
                count += 1

        day_pool = pd.concat(all_dfs)
        mkt_df = pd.read_csv(BREADTH_DATA_PATH, index_col='date', parse_dates=True)
        mkt_data = mkt_df.loc[:latest_date].tail(20)

        full_response = pm.get_daily_picks(day_pool, mkt_data)

        for p in full_response.get('predictions', []):
            if p['symbol'] == symbol_is:
                return {
                    "symbol": symbol_is,
                    "date": latest_date,
                    "selected": True,
                    "details": p
                }

        return {
            "symbol": symbol_is,
            "date": latest_date,
            "selected": False,
            "message": "Evaluated but rejected by portfolio utility constraints."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Symbol Prediction Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error evaluating symbol")
