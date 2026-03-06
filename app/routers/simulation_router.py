from fastapi import APIRouter
from app.app.simulation.demand_simulator import generate_daily_demand

router = APIRouter(prefix="/simulation", tags=["Simulation"])


@router.get("/demand")
def simulate_demand(
    product_id: str,
    days: int,
    base_demand: float,
    std_dev: float,
):
    data = generate_daily_demand(
        product_id=product_id,
        days=days,
        base_demand=base_demand,
        std_dev=std_dev,
    )

    return {
        "product_id": product_id,
        "days": days,
        "simulation": data,
    }