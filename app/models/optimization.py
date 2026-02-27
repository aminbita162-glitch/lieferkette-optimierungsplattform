from pydantic import BaseModel, Field
from typing import List, Optional


class DemandItem(BaseModel):
    product_id: str = Field(..., example="SKU-001")
    forecast_demand: float = Field(..., example=120.5)
    current_inventory: float = Field(..., example=80)

    demand_std: Optional[float] = Field(
        default=None,
        description="Demand standard deviation per day (same unit as forecast_demand per day).",
        example=15.0,
    )
    lead_time_days: Optional[float] = Field(
        default=None,
        description="Lead time in days.",
        example=7.0,
    )


class OptimizationInput(BaseModel):
    service_level: float = Field(..., ge=0.5, le=0.9999, example=0.95)
    holding_cost: float = Field(..., ge=0.0, example=2.5)
    shortage_cost: float = Field(..., ge=0.0, example=10.0)

    demand: List[DemandItem]

    default_demand_std: float = Field(
        default=10.0,
        ge=0.0,
        description="Fallback demand std per day if item.demand_std is not provided.",
        example=10.0,
    )
    default_lead_time_days: float = Field(
        default=7.0,
        gt=0.0,
        description="Fallback lead time in days if item.lead_time_days is not provided.",
        example=7.0,
    )


class OptimizationResultItem(BaseModel):
    product_id: str
    recommended_order_quantity: float
    safety_stock: float
    reorder_point: float
    expected_shortage: float
    total_cost_item: float


class OptimizationResult(BaseModel):
    total_cost: float
    recommendations: List[OptimizationResultItem]