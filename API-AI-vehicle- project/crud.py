from sqlalchemy.orm import Session
import models, schemas

def create_vehicle(db: Session, vehicle: schemas.VehicleCreate):
    """
    Creates a new vehicle log in the database.
    """
    db_vehicle = models.VehicleModel(
        tracking_id=vehicle.id,          
        vehicle_type=vehicle.vehicle_type,
        vehicle_img=vehicle.vehicle_img, 
        detection_time=vehicle.time
    )
    db.add(db_vehicle)
    db.commit()
    db.refresh(db_vehicle)
    return db_vehicle

def get_vehicle_count(db: Session):
    """
    Returns the total number of vehicles detected.
    """
    return db.query(models.VehicleModel).count()

def get_recent_vehicles(db: Session, limit: int = 5):
    """
    Fetches the most recently detected vehicles.
    """
    return db.query(models.VehicleModel).order_by(models.VehicleModel.detection_time.desc()).limit(limit).all()

def delete_vehicle(db: Session, vehicle_id: int):
    """
    Deletes a vehicle record by its ID.
    """
    vehicle_to_delete = db.query(models.VehicleModel).filter(models.VehicleModel.id == vehicle_id).first()
    if vehicle_to_delete:
        db.delete(vehicle_to_delete)
        db.commit()
        return True 
    return False