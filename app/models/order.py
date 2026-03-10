from sqlalchemy import Column, Integer, Float, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime

from app.database import Base


class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, index=True)

    description = Column(String, nullable=True)

    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)

    demand = Column(Integer, nullable=False)

    status = Column(String, default="pending", nullable=False)
    assigned_warehouse_name = Column(String, nullable=True)
    optimized_route = Column(String, nullable=True)

    warehouse_id = Column(Integer, ForeignKey("warehouses.id"), nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    warehouse = relationship("Warehouse")
    owner = relationship("User")