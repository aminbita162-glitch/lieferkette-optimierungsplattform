from sqlalchemy import Column, Integer, Float, String, ForeignKey
from sqlalchemy.orm import relationship

from app.database import Base


class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, index=True)

    description = Column(String, nullable=True)

    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)

    demand = Column(Integer, nullable=False)

    warehouse_id = Column(Integer, ForeignKey("warehouses.id"), nullable=True)

    owner_id = Column(Integer, ForeignKey("users.id"))

    warehouse = relationship("Warehouse")
    owner = relationship("User")