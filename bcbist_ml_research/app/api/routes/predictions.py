from fastapi import APIRouter, Request, HTTPException
import pandas as pd
from app.config import DATA_FEATURES_DIR

router = APIRouter()

@router.get("/daily-picks")
async def get_daily_picks(request: Request):
    pm = request.app.state.production_manager

    # In a real app, this would trigger the LiveDataProvider.
    # For now, we use the latest available feature data.
    try:
        # Mocking data load for the latest day
        latest_date = "2026-08-12" # Anchored to latest known test date
        all_dfs = []
        for f in DATA_FEATURES_DIR.glob("*.csv"):
            df = pd.read_csv(f, index_col='date', parse_dates=True)
            if latest_date in df.index:
                row = df.loc[[latest_date]].copy()
                row['symbol_col'] = f.stem.replace('_features', '').replace('_', '.')
                all_dfs.append(row)

        if not all_dfs:
            raise HTTPException(status_code=404, detail="No data available for prediction")

        day_data = pd.concat(all_dfs)
        market_data = pd.read_csv('data/market_breadth.csv', index_col='date', parse_dates=True).loc[[latest_date]]

        response = pm.get_daily_picks(day_data, market_data)
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{symbol}")
async def get_prediction(symbol: str):
    return {
        "symbol": symbol,
        "message": "Prediction model not trained yet",
        "as_of": None
    }
