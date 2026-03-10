from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.database import get_db
from app.models.warehouse import Warehouse
from app.models.user import User
from app.main import get_current_user


router = APIRouter(prefix="/warehouses", tags=["Warehouses"])


# -------------------------------------------------------------------
# Create warehouse
# -------------------------------------------------------------------
@router.post("/")
def create_warehouse(
    name: str,
    location: str,
    capacity: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):

    warehouse = Warehouse(
        name=name,
        location=location,
        capacity=capacity,
        owner_id=current_user.email
    )

    db.add(warehouse)
    db.commit()
    db.refresh(warehouse)

    return warehouse


# -------------------------------------------------------------------
# List warehouses
# -------------------------------------------------------------------
@router.get("/", response_model=List[dict])
def list_warehouses(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):

    warehouses = db.query(Warehouse).filter(
        Warehouse.owner_id == current_user.email
    ).all()

    return warehouses


# -------------------------------------------------------------------
# Get single warehouse
# -------------------------------------------------------------------
@router.get("/{warehouse_id}")
def get_warehouse(
    warehouse_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):

    warehouse = db.query(Warehouse).filter(
        Warehouse.id == warehouse_id,
        Warehouse.owner_id == current_user.email
    ).first()

    if not warehouse:
        raise HTTPException(status_code=404, detail="Warehouse not found")

    return warehouse


# -------------------------------------------------------------------
# Delete warehouse
# -------------------------------------------------------------------
@router.delete("/{warehouse_id}")
def delete_warehouse(
    warehouse_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):

    warehouse = db.query(Warehouse).filter(
        Warehouse.id == warehouse_id,
        Warehouse.owner_id == current_user.email
    ).first()

    if not warehouse:
        raise HTTPException(status_code=404, detail="Warehouse not found")

    db.delete(warehouse)
    db.commit()

    return {"message": "Warehouse deleted"}