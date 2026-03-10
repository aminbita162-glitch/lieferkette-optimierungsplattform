from fastapi import APIRouter
from app.services.demand_forecast import forecast_demand
from app.services.route_optimizer import optimize_route
from app.services.warehouse_ai import allocate_warehouse

router = APIRouter(prefix="/ai", tags=["AI Logistics"])


# ---------------------------------
# Demand Forecast Endpoint
# ---------------------------------
@router.post("/forecast")
def demand_forecast_endpoint(data: dict):

    history = data.get("history", [])
    result = forecast_demand(history)

    return {
        "history": history,
        "forecast": result
    }


# ---------------------------------
# Route Optimization Endpoint
# ---------------------------------
@router.post("/route-optimize")
def route_optimize_endpoint(data: dict):

    matrix = data.get("distance_matrix", [])
    route = optimize_route(matrix)

    return {
        "distance_matrix": matrix,
        "optimized_route": route
    }


# ---------------------------------
# Warehouse Allocation Endpoint
# ---------------------------------
@router.post("/warehouse-allocate")
def warehouse_allocate_endpoint(data: dict):

    order_location = data.get("order_location", {})
    warehouses = data.get("warehouses", [])

    result = allocate_warehouse(order_location, warehouses)

    return {
        "order_location": order_location,
        "warehouses": warehouses,
        "allocation": result
    }


# ---------------------------------
# AI Logistics Decision Engine
# ---------------------------------
@router.post("/logistics-plan")
def logistics_plan(data: dict):

    order_location = data.get("order_location", {})
    warehouses = data.get("warehouses", [])
    distance_matrix = data.get("distance_matrix", [])
    demand_history = data.get("demand_history", [])

    # 1️⃣ Forecast demand
    forecast = forecast_demand(demand_history)

    # 2️⃣ Choose best warehouse
    warehouse_result = allocate_warehouse(order_location, warehouses)

    # 3️⃣ Optimize delivery route
    route = optimize_route(distance_matrix)

    return {
        "forecast": forecast,
        "warehouse_selection": warehouse_result,
        "optimized_route": route
    }