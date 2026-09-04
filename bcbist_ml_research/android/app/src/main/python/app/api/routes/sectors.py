from fastapi import APIRouter
from app.config import STOCK_UNIVERSE

router = APIRouter()

@router.get("/")
async def get_sectors():
    sectors = sorted(list(set(s["sector"] for s in STOCK_UNIVERSE)))
    return {"sectors": sectors}

@router.get("/{sector}")
async def get_sector_info(sector: str):
    stocks = [s for s in STOCK_UNIVERSE if s["sector"].lower() == sector.lower()]
    return {"sector": sector, "stock_count": len(stocks), "stocks": stocks}
