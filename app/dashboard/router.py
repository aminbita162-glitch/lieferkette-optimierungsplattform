from fastapi import APIRouter
import requests

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


API_BASE = "https://lieferkette-optimierungsplattform.onrender.com"


@router.get("/status")
def dashboard_status():
    return {
        "status": "ok",
        "message": "Dashboard API is running"
    }


@router.get("/health")
def backend_health():
    r = requests.get(f"{API_BASE}/health")
    return r.json()


@router.get("/sample-optimization")
def sample_optimization():

    payload = {
        "service_level": 0.95,
        "holding_cost": 2.5,
        "shortage_cost": 10,
        "demand": [
            {
                "product_id": "SKU-001",
                "forecast_demand": 120.5,
                "current_inventory": 80,
                "demand_std": 15,
                "lead_time_days": 7
            }
        ],
        "default_demand_std": 10,
        "default_lead_time_days": 7
    }

    r = requests.post(
        f"{API_BASE}/optimize",
        json=payload
    )

    return r.json()