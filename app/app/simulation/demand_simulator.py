import random
from typing import List, Dict


def generate_daily_demand(
    product_id: str,
    days: int,
    base_demand: float,
    std_dev: float
) -> List[Dict]:

    results = []

    for day in range(1, days + 1):

        demand = max(0, random.gauss(base_demand, std_dev))

        results.append({
            "day": day,
            "product_id": product_id,
            "demand": round(demand, 2)
        })

    return results