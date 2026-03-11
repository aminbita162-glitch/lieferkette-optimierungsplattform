from collections import defaultdict
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.order import Order
from app.models.user import User
from app.models.warehouse import Warehouse
from app.services.route_optimizer import optimize_route

SECRET_KEY = "change-this-secret-key-in-production"
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

router = APIRouter(prefix="/routes", tags=["Routes"])


def euclidean_distance(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    return ((a_lat - b_lat) ** 2 + (a_lon - b_lon) ** 2) ** 0.5


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: Optional[str] = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.email == email.lower()).first()
    if user is None:
        raise credentials_exception

    return user


def build_distance_matrix(stops: List[Dict]) -> List[List[float]]:
    matrix: List[List[float]] = []

    for i in range(len(stops)):
        row = []
        for j in range(len(stops)):
            row.append(
                euclidean_distance(
                    stops[i]["latitude"],
                    stops[i]["longitude"],
                    stops[j]["latitude"],
                    stops[j]["longitude"],
                )
            )
        matrix.append(row)

    return matrix


@router.post("/optimize")
def optimize_routes(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    orders = db.query(Order).filter(
        Order.owner_id == current_user.id
    ).all()

    if not orders:
        return {
            "optimized_route": [],
            "route_groups": [],
            "message": "No orders available"
        }

    warehouses = db.query(Warehouse).filter(
        Warehouse.owner_id == current_user.id
    ).all()

    warehouse_by_id = {warehouse.id: warehouse for warehouse in warehouses}

    grouped_orders: Dict[str, List[Order]] = defaultdict(list)

    for order in orders:
        if order.warehouse_id is not None:
            group_key = f"warehouse_id:{order.warehouse_id}"
        elif order.assigned_warehouse_name:
            group_key = f"warehouse_name:{order.assigned_warehouse_name}"
        else:
            group_key = "unassigned"

        grouped_orders[group_key].append(order)

    route_groups = []
    optimized_route_flat = []

    for group_key, group_orders in grouped_orders.items():
        warehouse = None

        if group_key.startswith("warehouse_id:"):
            warehouse_id = int(group_key.split(":")[1])
            warehouse = warehouse_by_id.get(warehouse_id)
        elif group_key.startswith("warehouse_name:"):
            warehouse_name = group_key.split(":", 1)[1]
            warehouse = next(
                (w for w in warehouses if w.name == warehouse_name),
                None
            )

        stops: List[Dict] = []

        if warehouse is not None:
            stops.append(
                {
                    "type": "warehouse",
                    "id": warehouse.id,
                    "name": warehouse.name,
                    "latitude": warehouse.latitude,
                    "longitude": warehouse.longitude,
                }
            )

        for order in group_orders:
            stops.append(
                {
                    "type": "order",
                    "id": order.id,
                    "name": order.description or f"Order {order.id}",
                    "latitude": order.latitude,
                    "longitude": order.longitude,
                    "status": order.status,
                    "warehouse_id": order.warehouse_id,
                }
            )

        if not stops:
            continue

        distance_matrix = build_distance_matrix(stops)
        route_indexes = optimize_route(distance_matrix)
        ordered_stops = [stops[index] for index in route_indexes]

        total_distance = 0.0
        for i in range(len(route_indexes) - 1):
            total_distance += distance_matrix[route_indexes[i]][route_indexes[i + 1]]

        route_groups.append(
            {
                "group_key": group_key,
                "warehouse": {
                    "id": warehouse.id,
                    "name": warehouse.name,
                    "latitude": warehouse.latitude,
                    "longitude": warehouse.longitude,
                } if warehouse else None,
                "optimized_route_indexes": route_indexes,
                "ordered_stops": ordered_stops,
                "total_distance": total_distance,
            }
        )

        for stop in ordered_stops:
            if stop["type"] == "order":
                optimized_route_flat.append(stop["id"])

    return {
        "optimized_route": optimized_route_flat,
        "route_groups": route_groups,
    }