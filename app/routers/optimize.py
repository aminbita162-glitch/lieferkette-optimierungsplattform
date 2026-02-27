from __future__ import annotations

from typing import List
from math import erf, exp, pi, sqrt, log

from fastapi import APIRouter

from app.models.optimization import (
    OptimizationInput,
    OptimizationResult,
    OptimizationResultItem,
)

router = APIRouter(tags=["Optimize"])


def _phi(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def _Phi(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _inv_norm_cdf(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p must be in (0,1)")

    # Approximation (Acklam)
    a1 = -39.6968302866538
    a2 = 220.946098424521
    a3 = -275.928510446969
    a4 = 138.357751867269
    a5 = -30.6647980661472
    a6 = 2.50662827745924

    b1 = -54.4760987982241
    b2 = 161.585836858041
    b3 = -155.698979859887
    b4 = 66.8013118877197
    b5 = -13.2806815528857

    c1 = -0.00778489400243029
    c2 = -0.322396458041136
    c3 = -2.40075827716184
    c4 = -2.54973253934373
    c5 = 4.37466414146497
    c6 = 2.93816398269878

    d1 = 0.00778469570904146
    d2 = 0.32246712907004
    d3 = 2.445134137143
    d4 = 3.75440866190742

    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = sqrt(-2 * log(p))
        return (
            (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
        )

    if p > phigh:
        q = sqrt(-2 * log(1 - p))
        return -(
            (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
        )

    q = p - 0.5
    r = q * q

    return (
        (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
        (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1)
    )


@router.post("/optimize", response_model=OptimizationResult)
def optimize(payload: OptimizationInput) -> OptimizationResult:
    z = _inv_norm_cdf(payload.service_level)

    recommendations: List[OptimizationResultItem] = []
    total_cost = 0.0

    for item in payload.demand:
        d = item.forecast_demand
        inv = item.current_inventory

        sigma_d = item.demand_std if item.demand_std is not None else payload.default_demand_std
        lt = item.lead_time_days if item.lead_time_days is not None else payload.default_lead_time_days

        mean_lt = d * lt
        sigma_lt = sigma_d * sqrt(lt)

        safety_stock = z * sigma_lt
        reorder_point = mean_lt + safety_stock

        recommended_order_quantity = max(0.0, reorder_point - inv)

        expected_shortage = 0.0
        if sigma_lt > 0:
            k = (inv - mean_lt) / sigma_lt
            expected_shortage = sigma_lt * (_phi(k) - k * (1 - _Phi(k)))

        holding = payload.holding_cost
        shortage = payload.shortage_cost

        total_cost_item = holding * (safety_stock + 0.5 * recommended_order_quantity) + shortage * expected_shortage
        total_cost += total_cost_item

        recommendations.append(
            OptimizationResultItem(
                product_id=item.product_id,
                recommended_order_quantity=recommended_order_quantity,
                safety_stock=safety_stock,
                reorder_point=reorder_point,
                expected_shortage=expected_shortage,
                total_cost_item=total_cost_item,
            )
        )

    return OptimizationResult(
        total_cost=total_cost,
        recommendations=recommendations,
    )