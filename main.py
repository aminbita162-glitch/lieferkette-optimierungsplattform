from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
import numpy as np

app = FastAPI(
    title="Lieferkette Optimierungsplattform API",
    version="0.2.0",
    description="Supply Chain Optimization & Simulation API",
)

# -------------------------
# Data Models
# -------------------------

class OptimizationInput(BaseModel):
    demand: List[float] = Field(..., example=[100, 150, 200])
    supply: List[float] = Field(..., example=[120, 180, 160])
    cost_per_unit: float = Field(..., example=5.0)


class OptimizationResult(BaseModel):
    total_demand: float
    total_supply: float
    fulfilled: float
    shortage: float
    total_cost: float
    efficiency_score: float


# -------------------------
# Root
# -------------------------

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "API is running",
        "docs": "/docs",
        "redoc": "/redoc",
    }


@app.get("/health")
def health():
    return {"ok": True}


# -------------------------
# Optimization Endpoint
# -------------------------

@app.post("/optimize", response_model=OptimizationResult)
def optimize(data: OptimizationInput):

    total_demand = float(np.sum(data.demand))
    total_supply = float(np.sum(data.supply))

    fulfilled = min(total_demand, total_supply)
    shortage = max(0.0, total_demand - total_supply)

    total_cost = fulfilled * data.cost_per_unit

    efficiency_score = (
        fulfilled / total_demand if total_demand > 0 else 0
    )

    return OptimizationResult(
        total_demand=total_demand,
        total_supply=total_supply,
        fulfilled=fulfilled,
        shortage=shortage,
        total_cost=total_cost,
        efficiency_score=round(efficiency_score, 4),
    )