from __future__ import annotations

from typing import List
from math import erf, exp, pi, sqrt, log

from fastapi import APIRouter, HTTPException, status

from app.models.optimization import (
    OptimizationInput,
    OptimizationResult,
    OptimizationResultItem,
)

router = APIRouter(tags=["Optimize"])


# ---------------------------
# Helpers (math)
# ---------------------------
def _phi(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def _Phi(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _inv_norm_cdf(p: float) -> float:
    """
    Inverse CDF (quantile) of standard normal.
    Uses Peter J. Acklam approximation.
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError("service_level must be in (0, 1)")

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
            (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
            / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
        )

    if p > phigh:
        q = sqrt(-2 * log(1 - p))
        return -(
            (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
            / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
        )

    q = p - 0.5
    r = q * q

    return (
        (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
        * q
        / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1)
    )


# ---------------------------
# Helpers (API error format)
# ---------------------------
def _api_error(code: str, message: str, *, details: dict | None = None) -> dict:
    """
    Unified error payload (we put it inside HTTPException.detail)
    """
    payload = {
        "error": {
            "code": code,
            "message": message,
        }
    }
    if details:
        payload["error"]["details"] = details
    return payload


def _bad_request(code: str, message: str, *, details: dict | None = None) -> None:
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=_api_error(code, message, details=details),
    )


@router.post(
    "/optimize",
    response_model=OptimizationResult,
    responses={
        400: {
            "description": "Bad Request (domain / business validation)",
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "code": "INVALID_INPUT",
                            "message": "service_level must be in (0, 1)",
                            "details": {"field": "service_level"},
                        }
                    }
                }
            },
        },
        422: {"description": "Validation Error (schema)"},
        500: {"description": "Internal Server Error"},
    },
)
def optimize(payload: OptimizationInput) -> OptimizationResult:
    # ---- Domain validations (we keep them consistent with 400 errors)
    if payload.holding_cost < 0:
        _bad_request("INVALID_INPUT", "holding_cost must be >= 0", details={"field": "holding_cost"})
    if payload.shortage_cost < 0:
        _bad_request("INVALID_INPUT", "shortage_cost must be >= 0", details={"field": "shortage_cost"})

    if payload.default_demand_std is None or payload.default_demand_std < 0:
        _bad_request(
            "INVALID_INPUT",
            "default_demand_std must be provided and >= 0",
            details={"field": "default_demand_std"},
        )
    if payload.default_lead_time_days is None or payload.default_lead_time_days <= 0:
        _bad_request(
            "INVALID_INPUT",
            "default_lead_time_days must be provided and > 0",
            details={"field": "default_lead_time_days"},
        )

    try:
        z = _inv_norm_cdf(payload.service_level)
    except ValueError:
        _bad_request(
            "INVALID_INPUT",
            "service_level must be in (0, 1)",
            details={"field": "service_level"},
        )

    recommendations: List[OptimizationResultItem] = []
    total_cost = 0.0

    for idx, item in enumerate(payload.demand):
        # Per-item domain checks
        if item.forecast_demand < 0:
            _bad_request(
                "INVALID_INPUT",
                "forecast_demand must be >= 0",
                details={"item_index": idx, "field": "forecast_demand", "product_id": item.product_id},
            )
        if item.current_inventory < 0:
            _bad_request(
                "INVALID_INPUT",
                "current_inventory must be >= 0",
                details={"item_index": idx, "field": "current_inventory", "product_id": item.product_id},
            )
        if item.demand_std is not None and item.demand_std < 0:
            _bad_request(
                "INVALID_INPUT",
                "demand_std must be >= 0",
                details={"item_index": idx, "field": "demand_std", "product_id": item.product_id},
            )
        if item.lead_time_days is not None and item.lead_time_days <= 0:
            _bad_request(
                "INVALID_INPUT",
                "lead_time_days must be > 0",
                details={"item_index": idx, "field": "lead_time_days", "product_id": item.product_id},
            )

        d = float(item.forecast_demand)
        inv = float(item.current_inventory)

        sigma_d = float(item.demand_std) if item.demand_std is not None else float(payload.default_demand_std)
        lt = float(item.lead_time_days) if item.lead_time_days is not None else float(payload.default_lead_time_days)

        # Mean & std during lead time
        mean_lt = d * lt
        sigma_lt = sigma_d * sqrt(lt) if lt > 0 else 0.0

        safety_stock = z * sigma_lt if sigma_lt > 0 else 0.0
        reorder_point = mean_lt + safety_stock

        recommended_order_quantity = max(0.0, reorder_point - inv)

        expected_shortage = 0.0
        if sigma_lt > 0:
            k = (inv - mean_lt) / sigma_lt
            expected_shortage = sigma_lt * (_phi(k) - k * (1 - _Phi(k)))

        holding = float(payload.holding_cost)
        shortage = float(payload.shortage_cost)

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