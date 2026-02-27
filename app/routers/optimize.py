from fastapi import APIRouter
from app.models.optimization import (
    OptimizationInput,
    OptimizationResult,
    OptimizationResultItem,
)

router = APIRouter(tags=["Optimize"])


@router.post("/optimize", response_model=OptimizationResult)
def optimize(payload: OptimizationInput):
    recs = []
    total_cost = 0.0

    for item in payload.demand:
        # Safety stock (نسخه MVP ساده)
        safety_stock = max(
            item.forecast_demand * (1.0 - payload.service_level), 0.0
        )

        # Recommended order quantity
        order_qty = max(
            item.forecast_demand - item.current_inventory + safety_stock, 0.0
        )

        # Simple cost estimation
        holding = payload.holding_cost * max(
            item.current_inventory + order_qty - item.forecast_demand, 0.0
        )

        shortage = payload.shortage_cost * max(
            item.forecast_demand - (item.current_inventory + order_qty), 0.0
        )

        total_cost += holding + shortage

        recs.append(
            OptimizationResultItem(
                product_id=item.product_id,
                recommended_order_quantity=round(order_qty, 4),
                safety_stock=round(safety_stock, 4),
            )
        )

    return OptimizationResult(
        total_cost=round(total_cost, 4),
        recommendations=recs,
    )