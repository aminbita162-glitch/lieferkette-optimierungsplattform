from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.order import Order
from app.models.user import User
from app.models.warehouse import Warehouse
from app.services.warehouse_ai import allocate_warehouse


SECRET_KEY = "change-this-secret-key-in-production"
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

router = APIRouter(prefix="/orders", tags=["Orders"])


class OrderCreate(BaseModel):
    description: Optional[str] = None
    latitude: float
    longitude: float
    demand: int
    warehouse_id: Optional[int] = None


class OrderResponse(BaseModel):
    id: int
    description: Optional[str] = None
    latitude: float
    longitude: float
    demand: int
    status: str
    assigned_warehouse_name: Optional[str] = None
    optimized_route: Optional[str] = None
    warehouse_id: Optional[int] = None
    owner_id: int
    created_at: datetime

    class Config:
        from_attributes = True


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


@router.post("/", response_model=OrderResponse, status_code=status.HTTP_201_CREATED)
def create_order(
    payload: OrderCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    selected_warehouse = None
    selected_warehouse_name = None
    selected_warehouse_id = payload.warehouse_id
    status_value = "pending"

    if payload.warehouse_id is not None:
        selected_warehouse = db.query(Warehouse).filter(
            Warehouse.id == payload.warehouse_id,
            Warehouse.owner_id == current_user.id
        ).first()

        if not selected_warehouse:
            raise HTTPException(status_code=404, detail="Warehouse not found")

        selected_warehouse_name = selected_warehouse.name
        status_value = "planned"

    else:
        user_warehouses = db.query(Warehouse).filter(
            Warehouse.owner_id == current_user.id
        ).all()

        if user_warehouses:
            warehouse_candidates = [
                {
                    "id": w.id,
                    "name": w.name,
                    "latitude": w.latitude,
                    "longitude": w.longitude,
                }
                for w in user_warehouses
            ]

            allocation = allocate_warehouse(
                {
                    "latitude": payload.latitude,
                    "longitude": payload.longitude,
                },
                warehouse_candidates,
            )

            selected_warehouse = allocation.get("selected_warehouse")
            if selected_warehouse:
                selected_warehouse_name = selected_warehouse.get("name")
                selected_warehouse_id = selected_warehouse.get("id")
                status_value = "planned"

    order = Order(
        description=payload.description,
        latitude=payload.latitude,
        longitude=payload.longitude,
        demand=payload.demand,
        status=status_value,
        assigned_warehouse_name=selected_warehouse_name,
        optimized_route=None,
        warehouse_id=selected_warehouse_id,
        owner_id=current_user.id,
    )

    db.add(order)
    db.commit()
    db.refresh(order)

    return order


@router.get("/", response_model=List[OrderResponse])
def list_orders(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    orders = db.query(Order).filter(
        Order.owner_id == current_user.id
    ).all()

    return orders


@router.get("/{order_id}", response_model=OrderResponse)
def get_order(
    order_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    order = db.query(Order).filter(
        Order.id == order_id,
        Order.owner_id == current_user.id
    ).first()

    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    return order


@router.delete("/{order_id}")
def delete_order(
    order_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    order = db.query(Order).filter(
        Order.id == order_id,
        Order.owner_id == current_user.id
    ).first()

    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    db.delete(order)
    db.commit()

    return {"message": "Order deleted"}