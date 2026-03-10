from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User
from app.models.warehouse import Warehouse


SECRET_KEY = "change-this-secret-key-in-production"
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

router = APIRouter(prefix="/warehouses", tags=["Warehouses"])


class WarehouseCreate(BaseModel):
    name: str
    latitude: float
    longitude: float
    capacity: int


class WarehouseResponse(BaseModel):
    id: int
    name: str
    latitude: float
    longitude: float
    capacity: int
    owner_id: Optional[int] = None

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


@router.post("/", response_model=WarehouseResponse, status_code=status.HTTP_201_CREATED)
def create_warehouse(
    payload: WarehouseCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    warehouse = Warehouse(
        name=payload.name,
        latitude=payload.latitude,
        longitude=payload.longitude,
        capacity=payload.capacity,
        owner_id=current_user.id
    )

    db.add(warehouse)
    db.commit()
    db.refresh(warehouse)

    return warehouse


@router.get("/", response_model=List[WarehouseResponse])
def list_warehouses(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    warehouses = db.query(Warehouse).filter(
        Warehouse.owner_id == current_user.id
    ).all()

    return warehouses


@router.get("/{warehouse_id}", response_model=WarehouseResponse)
def get_warehouse(
    warehouse_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    warehouse = db.query(Warehouse).filter(
        Warehouse.id == warehouse_id,
        Warehouse.owner_id == current_user.id
    ).first()

    if not warehouse:
        raise HTTPException(status_code=404, detail="Warehouse not found")

    return warehouse


@router.delete("/{warehouse_id}")
def delete_warehouse(
    warehouse_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    warehouse = db.query(Warehouse).filter(
        Warehouse.id == warehouse_id,
        Warehouse.owner_id == current_user.id
    ).first()

    if not warehouse:
        raise HTTPException(status_code=404, detail="Warehouse not found")

    db.delete(warehouse)
    db.commit()

    return {"message": "Warehouse deleted"}