from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field


class DemandItem(BaseModel):
    product_id: str = Field(..., example="SKU-001")
    forecast_demand: float = Field(..., example=120.5)
    current_inventory: float = Field(..., example=80)


class OptimizationInput(BaseModel):
    service_level: float = Field(..., example=0.95)
    holding_cost: float = Field(..., example=2.5)
    shortage_cost: float = Field(..., example=10.0)
    demand: List[DemandItem]


class OptimizationResultItem(BaseModel):
    product_id: str
    recommended_order_quantity: float
    safety_stock: float


class OptimizationResult(BaseModel):
    total_cost: float
    recommendations: List[OptimizationResultItem]