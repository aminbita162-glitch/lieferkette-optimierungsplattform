from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.order import Order
from app.services.route_optimizer import optimize_route

router = APIRouter(prefix="/routes", tags=["Routes"])


def euclidean_distance(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    return ((a_lat - b_lat) ** 2 + (a_lon - b_lon) ** 2) ** 0.5


@router.post("/optimize")
def optimize_routes(db: Session = Depends(get_db)):
    orders = db.query(Order).all()

    if not orders:
        return {
            "optimized_route": [],
            "message": "No orders available"
        }

    locations = []
    for order in orders:
        locations.append((order.latitude, order.longitude))

    n = len(locations)
    distance_matrix = []

    for i in range(n):
        row = []
        for j in range(n):
            dist = euclidean_distance(
                locations[i][0],
                locations[i][1],
                locations[j][0],
                locations[j][1],
            )
            row.append(dist)
        distance_matrix.append(row)

    route = optimize_route(distance_matrix)
    ordered_orders = [orders[i].id for i in route]

    return {
        "optimized_route": ordered_orders
    }