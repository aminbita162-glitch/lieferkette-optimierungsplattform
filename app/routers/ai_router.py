from typing import Any, Dict, List

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.demand_forecast import forecast_demand
from app.services.route_optimizer import optimize_route
from app.services.warehouse_ai import allocate_warehouse


router = APIRouter(prefix="/ai", tags=["AI Logistics"])


class ForecastRequest(BaseModel):
    history: List[float] = Field(default_factory=list)


class RouteOptimizeRequest(BaseModel):
    distance_matrix: List[List[float]] = Field(default_factory=list)
    stop_labels: List[str] = Field(default_factory=list)


class WarehouseAllocateRequest(BaseModel):
    order_location: Dict[str, float] = Field(default_factory=dict)
    warehouses: List[Dict[str, Any]] = Field(default_factory=list)


class LogisticsPlanRequest(BaseModel):
    order_location: Dict[str, float] = Field(default_factory=dict)
    warehouses: List[Dict[str, Any]] = Field(default_factory=list)
    distance_matrix: List[List[float]] = Field(default_factory=list)
    stop_labels: List[str] = Field(default_factory=list)
    demand_history: List[float] = Field(default_factory=list)


@router.post("/forecast")
def demand_forecast_endpoint(payload: ForecastRequest):
    result = forecast_demand(payload.history)

    return {
        "history": payload.history,
        "forecast": result
    }


@router.post("/route-optimize")
def route_optimize_endpoint(payload: RouteOptimizeRequest):
    route_indexes = optimize_route(payload.distance_matrix)

    labeled_route = []
    if payload.stop_labels and len(payload.stop_labels) == len(route_indexes):
        try:
            labeled_route = [payload.stop_labels[index] for index in route_indexes]
        except Exception:
            labeled_route = []

    return {
        "distance_matrix": payload.distance_matrix,
        "optimized_route_indexes": route_indexes,
        "optimized_route_labels": labeled_route
    }


@router.post("/warehouse-allocate")
def warehouse_allocate_endpoint(payload: WarehouseAllocateRequest):
    result = allocate_warehouse(payload.order_location, payload.warehouses)

    return {
        "order_location": payload.order_location,
        "warehouses": payload.warehouses,
        "allocation": result
    }


@router.post("/logistics-plan")
def logistics_plan(payload: LogisticsPlanRequest):
    forecast = forecast_demand(payload.demand_history)
    warehouse_result = allocate_warehouse(payload.order_location, payload.warehouses)
    route_indexes = optimize_route(payload.distance_matrix)

    labeled_route = []
    if payload.stop_labels and len(payload.stop_labels) == len(route_indexes):
        try:
            labeled_route = [payload.stop_labels[index] for index in route_indexes]
        except Exception:
            labeled_route = []

    return {
        "forecast": forecast,
        "warehouse_selection": warehouse_result,
        "optimized_route_indexes": route_indexes,
        "optimized_route_labels": labeled_route
    }