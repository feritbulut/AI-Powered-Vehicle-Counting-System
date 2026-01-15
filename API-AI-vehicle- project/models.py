from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.sql import func
from database import Base

class VehicleModel(Base):
    
    __tablename__ = "vehicle_logs_tables"

    id = Column(Integer, primary_key=True, index=True)
    
    # Metadata from AI
    tracking_id = Column(String(100))  # Original ID from the tracking algorithm
    vehicle_type = Column(String(50))  # e.g., Car, Truck, Bus
    vehicle_img = Column(Text)         # Base64 encoded image string
    
    # Timestamps
    detection_time = Column(DateTime)  # Time sent by the AI unit
    created_at = Column(DateTime(timezone=True), server_default=func.now()) # DB insertion time