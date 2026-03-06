from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


class DomainValidationError(ValueError):
    """Business/domain validation error."""


def _require(cond: bool, msg: str, field: Optional[str] = None) -> None:
    if not cond:
        if field:
            raise DomainValidationError(f"{msg} (field={field})")
        raise DomainValidationError(msg)


def _to_float(x: Any, field: str) -> float:
    try:
        v = float(x)
    except Exception as exc:
        raise DomainValidationError(f"Invalid number for {field}") from exc
    if math.isnan(v) or math.isinf(v):
        raise DomainValidationError(f"Invalid number for {field}")
    return v


def _to_int(x: Any, field: str) -> int:
    try:
        v = int(x)
    except Exception as exc:
        raise DomainValidationError(f"Invalid integer for {field}") from exc
    return v


def optimize(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Core optimization logic (simple, deterministic, fast).

    Expected payload (JSON):
      {
        "service_level": 0.95,                # (0,1)
        "holding_cost": 2.5,                  # >= 0
        "shortage_cost": 10,                  # >= 0
        "demand": [
          {
            "product_id": "SKU-001",
            "forecast_demand": 120.5,         # >= 0
            "current_inventory": 80,          # >= 0
            "demand_std": 15,                 # >= 0 (optional)
            "lead_time_days": 7               # >= 0 (optional)
          }
        ],
        "default_demand_std": 10,             # >= 0 (optional)
        "default_lead_time_days": 7           # >= 0 (optional)
      }
    """
    _require(isinstance(payload, dict), "Payload must be an object")

    service_level = _to_float(payload.get("service_level"), "service_level")
    holding_cost = _to_float(payload.get("holding_cost"), "holding_cost")
    shortage_cost = _to_float(payload.get("shortage_cost"), "shortage_cost")

    _require(0.0 < service_level < 1.0, "service_level must be in (0, 1)", "service_level")
    _require(holding_cost >= 0.0, "holding_cost must be >= 0", "holding_cost")
    _require(shortage_cost >= 0.0, "shortage_cost must be >= 0", "shortage_cost")

    default_demand_std = _to_float(payload.get("default_demand_std", 0.0), "default_demand_std")
    default_lead_time = _to_int(payload.get("default_lead_time_days", 0), "default_lead_time_days")
    _require(default_demand_std >= 0.0, "default_demand_std must be >= 0", "default_demand_std")
    _require(default_lead_time >= 0, "default_lead_time_days must be >= 0", "default_lead_time_days")

    demand_list = payload.get("demand", [])
    _require(isinstance(demand_list, list), "demand must be a list", "demand")
    _require(len(demand_list) > 0, "demand list must not be empty", "demand")

    recommendations: List[Dict[str, Any]] = []
    total_cost = 0.0

    # Normal-approx safety factor (z) for common service levels (kept simple)
    # If service_level not in map, approximate using inverse error function.
    z_map = {
        0.80: 0.8416,
        0.85: 1.0364,
        0.90: 1.2816,
        0.95: 1.6449,
        0.97: 1.8808,
        0.98: 2.0537,
        0.99: 2.3263,
    }

    def z_value(sl: float) -> float:
        key = round(sl, 2)
        if key in z_map:
            return z_map[key]
        # fallback approximation (good enough for this project level)
        # Abramowitz/Stegun style approximation via erfinv-like polynomial
        # Convert CDF to erf argument: p = 2*sl - 1
        p = 2.0 * sl - 1.0
        _require(-1.0 < p < 1.0, "service_level out of range", "service_level")
        # Approximate inverse erf
        a = 0.147
        ln = math.log(1.0 - p * p)
        t = 2.0 / (math.pi * a) + ln / 2.0
        erfinv = math.copysign(math.sqrt(max(0.0, math.sqrt(t * t - ln / a) - t)), p)
        # z = sqrt(2) * erfinv
        return math.sqrt(2.0) * erfinv

    z = z_value(service_level)

    for i, item in enumerate(demand_list):
        _require(isinstance(item, dict), f"demand[{i}] must be an object")

        product_id = item.get("product_id")
        _require(isinstance(product_id, str) and product_id.strip() != "", "product_id is required", "product_id")

        forecast = _to_float(item.get("forecast_demand", 0.0), "forecast_demand")
        inventory = _to_float(item.get("current_inventory", 0.0), "current_inventory")

        demand_std = item.get("demand_std", default_demand_std)
        lead_time = item.get("lead_time_days", default_lead_time)

        demand_std = _to_float(demand_std, "demand_std")
        lead_time = _to_int(lead_time, "lead_time_days")

        _require(forecast >= 0.0, "forecast_demand must be >= 0", "forecast_demand")
        _require(inventory >= 0.0, "current_inventory must be >= 0", "current_inventory")
        _require(demand_std >= 0.0, "demand_std must be >= 0", "demand_std")
        _require(lead_time >= 0, "lead_time_days must be >= 0", "lead_time_days")

        # --- Simple policy ---
        # reorder_point = mean demand during lead time + safety stock
        mean_lt = forecast * float(lead_time)
        safety_stock = z * demand_std * math.sqrt(max(1.0, float(lead_time)))
        reorder_point = mean_lt + safety_stock

        # recommended order if inventory is below reorder_point
        recommended_order_qty = max(0.0, reorder_point - inventory)

        # expected shortage (very simplified) if inventory << mean_lt
        expected_shortage = max(0.0, mean_lt - inventory)

        # cost proxy
        cost_item = holding_cost * safety_stock + shortage_cost * expected_shortage
        total_cost += cost_item

        recommendations.append(
            {
                "product_id": product_id,
                "recommended_order_quantity": float(recommended_order_qty),
                "safety_stock": float(safety_stock),
                "reorder_point": float(reorder_point),
                "expected_shortage": float(expected_shortage),
                "total_cost_item": float(cost_item),
            }
        )

    return {"total_cost": float(total_cost), "recommendations": recommendations}