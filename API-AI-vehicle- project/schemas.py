from pydantic import BaseModel
from datetime import datetime

class VehicleCreate(BaseModel):
    """
    Schema for incoming data from the AI Camera Unit.
    """
    id: str               # Tracking ID from AI
    vehicle_type: str
    vehicle_img: str      # Base64 string
    time: datetime        # Timestamp of detection

class VehicleResponse(BaseModel):
    """
    Schema for outgoing data (API responses).
    """
    id: int               # Database Primary Key
    tracking_id: str      
    vehicle_type: str
    detection_time: datetime
    created_at: datetime
    
    class Config:
        from_attributes = True