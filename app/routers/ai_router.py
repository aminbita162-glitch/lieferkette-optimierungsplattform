from fastapi import APIRouter
from app.services.demand_forecast import forecast_demand
from app.services.route_optimizer import optimize_route

router = APIRouter(prefix="/ai", tags=["AI Logistics"])


@router.post("/forecast")
def demand_forecast_endpoint(data: dict):
    history = data.get("history", [])
    result = forecast_demand(history)

    return {
        "history": history,
        "forecast": result
    }


@router.post("/route-optimize")
def route_optimize_endpoint(data: dict):
    matrix = data.get("distance_matrix", [])
    route = optimize_route(matrix)

    return {
        "distance_matrix": matrix,
        "optimized_route": route
    }