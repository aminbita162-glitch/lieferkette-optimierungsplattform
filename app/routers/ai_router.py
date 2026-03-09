from fastapi import APIRouter
from app.services.demand_forecast import forecast_demand

router = APIRouter(prefix="/ai", tags=["AI Logistics"])


@router.post("/forecast")
def demand_forecast_endpoint(data: dict):

    history = data.get("history", [])

    result = forecast_demand(history)

    return {
        "history": history,
        "forecast": result
    }