from __future__ import annotations

from typing import List
from fastapi import APIRouter
from app.models.optimization import OptimizationInput, OptimizationResult, OptimizationResultItem

router = APIRouter(tags=["Optimize"])

@router.post("/optimize", response_model=OptimizationResult)
def optimize(payload: OptimizationInput) -> OptimizationResult:
    """
    Simple baseline optimizer:
    - Allocates available stock to each item up to demand.
    - Computes unmet demand (shortage).
    """

    results: List[OptimizationResultItem] = []
    total_shortage = 0.0

    for item in payload.items:
        demand = float(item.demand)
        stock = float(item.current_stock)

        allocated = min(demand, stock)
        shortage = max(0.0, demand - stock)

        total_shortage += shortage

        results.append(
            OptimizationResultItem(
                sku=item.sku,
                demand=demand,
                current_stock=stock,
                allocated=allocated,
                shortage=shortage,
            )
        )

    return OptimizationResult(
        ok=True,
        objective="min_shortage_baseline",
        total_shortage=total_shortage,
        items=results,
    )