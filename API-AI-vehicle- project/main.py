import os
import json
from fastapi import FastAPI, Depends, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import models, schemas, crud
from database import engine, SessionLocal
from websocket_manager import manager

# Create Database Tables
models.Base.metadata.create_all(bind=engine)

# Initialize FastAPI
app = FastAPI(
    title="AI Vehicle Count System", 
    description="Backend API for Real-Time Vehicle Detection and Monitoring",
    version="2.0.0"
)

# CORS Configuration (Allow all origins for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency: Database Session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket Endpoint for Real-Time Communication.
    """
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/")
def read_root():
    return {"status": "System Online", "ready_for": "AI & Mobile Integration"}

@app.post("/api/detect", response_model=schemas.VehicleResponse)
async def detect_vehicle(data: schemas.VehicleCreate, db: Session = Depends(get_db)):
    """
    Receives detection data from the AI Unit, saves it to DB, 
    and broadcasts it to connected clients via WebSocket.
    """
    # 1. Save to Database
    saved_vehicle = crud.create_vehicle(db, data)
    
    # 2. Get Updated Stats
    total_count = crud.get_vehicle_count(db)
    
    # 3. Prepare Real-Time Payload
    realtime_data = {
        "type": "new_detection",
        "count": total_count,
        "vehicle_type": data.vehicle_type,
        "tracking_id": data.id,
        "image": data.vehicle_img, 
        "time": str(data.time)
    }
    
    # 4. Broadcast via WebSocket
    await manager.broadcast(json.dumps(realtime_data))
    
    return saved_vehicle

@app.get("/api/stats")
def get_stats(db: Session = Depends(get_db)):
    """
    Returns current statistics: Total count and recent logs.
    """
    return {
        "total_vehicles": crud.get_vehicle_count(db),
        "recent_logs": crud.get_recent_vehicles(db)
    }

@app.delete("/api/delete/{vehicle_id}")
async def delete_vehicle_endpoint(vehicle_id: int, db: Session = Depends(get_db)):
    """
    Deletes a vehicle record and notifies clients to update their UI.
    """
    if crud.delete_vehicle(db, vehicle_id):
        new_count = crud.get_vehicle_count(db)
        
        # Broadcast deletion event
        await manager.broadcast(json.dumps({
            "type": "deletion", 
            "count": new_count, 
            "deleted_id": vehicle_id
        }))
        
        return {"message": "Vehicle deleted successfully", "new_count": new_count}
    
    raise HTTPException(status_code=404, detail="Vehicle not found")

if __name__ == "__main__":
    import uvicorn
    # Get port from environment variable (default 8000)
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)