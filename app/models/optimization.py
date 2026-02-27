from __future__ import annotations

from typing import List

from fastapi import APIRouter
from app.models.optimization import (
    OptimizationInput,
    OptimizationResult,
    OptimizationResultItem,
)

router = APIRouter(tags=["Optimize"])


@router.post("/optimize", response_model=OptimizationResult)
def optimize(payload: OptimizationInput) -> OptimizationResult:
    recommendations: List[OptimizationResultItem] = []
    total_cost = 0.0

    for item in payload.demand:
        demand = float(item.forecast_demand)
        inventory = float(item.current_inventory)

        recommended_order_quantity = max(0.0, demand - inventory)
        safety_stock = 0.0

        total_cost += float(payload.holding_cost) * (recommended_order_quantity + safety_stock)

        recommendations.append(
            OptimizationResultItem(
                product_id=item.product_id,
                recommended_order_quantity=recommended_order_quantity,
                safety_stock=safety_stock,
            )
        )

    return OptimizationResult(
        total_cost=total_cost,
        recommendations=recommendations,
    )