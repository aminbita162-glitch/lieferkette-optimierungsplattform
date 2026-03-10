from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.database import get_db
from app.models.order import Order
from app.models.user import User
from app.main import get_current_user


router = APIRouter(prefix="/orders", tags=["Orders"])


# ------------------------------------------------
# Create Order
# ------------------------------------------------
@router.post("/")
def create_order(
    product_name: str,
    quantity: int,
    destination_lat: float,
    destination_lon: float,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):

    order = Order(
        product_name=product_name,
        quantity=quantity,
        destination_lat=destination_lat,
        destination_lon=destination_lon,
        owner_id=current_user.email
    )

    db.add(order)
    db.commit()
    db.refresh(order)

    return order


# ------------------------------------------------
# List Orders
# ------------------------------------------------
@router.get("/", response_model=List[dict])
def list_orders(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):

    orders = db.query(Order).filter(
        Order.owner_id == current_user.email
    ).all()

    return orders


# ------------------------------------------------
# Get Single Order
# ------------------------------------------------
@router.get("/{order_id}")
def get_order(
    order_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):

    order = db.query(Order).filter(
        Order.id == order_id,
        Order.owner_id == current_user.email
    ).first()

    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    return order


# ------------------------------------------------
# Delete Order
# ------------------------------------------------
@router.delete("/{order_id}")
def delete_order(
    order_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):

    order = db.query(Order).filter(
        Order.id == order_id,
        Order.owner_id == current_user.email
    ).first()

    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    db.delete(order)
    db.commit()

    return {"message": "Order deleted"}