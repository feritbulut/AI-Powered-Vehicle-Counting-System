# AI-Powered Vehicle Counting System

End-to-end platform for detecting, tracking, and counting vehicles in video streams. The system couples an AI vision pipeline (YOLOv8 + multiple trackers), a FastAPI backend for real-time ingestion and WebSocket push, and a .NET MAUI mobile client for monitoring.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![C#](https://img.shields.io/badge/C%23-239120?style=for-the-badge&logo=csharp&logoColor=white)
![.NET](https://img.shields.io/badge/.NET-512BD4?style=for-the-badge&logo=dotnet&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)

## Repository Map

- AracSayar Detection/ — Python vision pipeline (YOLOv8 + DeepSORT/IOU-Kalman/SORT) for detection, tracking, counting, overlay, and optional event posting.
- API-AI-vehicle- project/ — FastAPI service exposing REST + WebSocket endpoints, persisting detections to MySQL, and broadcasting real-time events.
- AracSayar Mobile/ — .NET MAUI application showing live vehicle updates and historical stats.

## How the System Flows

1. Camera or video feed enters the detection pipeline.
2. YOLOv8 detects vehicles; chosen tracker assigns stable IDs and maintains trajectories.
3. Counting controller tallies vehicles crossing virtual lines; crops and metadata are sent to the backend (HTTP) when enabled.
4. FastAPI stores events in MySQL and broadcasts them to connected mobile clients via WebSocket.
5. Mobile app renders live entries and stats pulled from the backend.

<p align="center">
  <img src="https://github.com/user-attachments/assets/0e2d4258-f0d4-49d6-97c6-32aaa9b1b9da" width="150"/>
  <img src="https://github.com/user-attachments/assets/de2b669d-8c44-418a-9595-e5497a6f0288" width="150"/>
</p>

## Quickstart (Local, All Components)

### Prerequisites

- Python 3.10+
- .NET 9 SDK for the MAUI app
- MySQL instance (local or cloud) for the backend

### 1) Vision Pipeline

```bash
cd "AracSayar Detection"
python -m venv .venv && source .venv/bin/activate
pip install -r deep_sort/requirements.txt
pip install ultralytics
python main.py --source testVideo.mp4 --display --tracker deepsort --db-endpoint http://localhost:8000/api/detect
```

Key flags: --source (camera index or video path), --tracker (deepsort, ioukalman, sort), --count-mode (single, double), --orientation (auto, horizontal, vertical).

### 2) Backend API

```bash
cd "API-AI-vehicle- project"
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export DATABASE_URL=mysql+pymysql://user:pass@host:3306/dbname
uvicorn main:app --reload --port 8000
```

Endpoints:
- POST /api/detect — ingest detection events (ID, type, image/crop, timestamp).
- WS /ws — broadcast stream for mobile clients.
- GET /api/stats — aggregated counts and recent entries.
- DELETE /api/delete/{id} — remove a record.

Interactive docs: visit http://localhost:8000/docs once running.

### 3) Mobile App

```bash
cd "AracSayar Mobile"
dotnet restore
dotnet build
# Run on chosen target, e.g. Android emulator
dotnet build -t:Run -f net9.0-android
```

Configure backend URL inside the app settings or constants before running.

## Notable Internals

- Detection/Tracking core: AracSayar Detection/src/pipeline.py, with interchangeable tracker adapters under src/tracking/ and YOLOv8 wrapper in src/detectors/yolo.py.
- Backend models and schemas: API-AI-vehicle- project/models.py and schemas.py; WebSocket handling in websocket_manager.py.
- Mobile data models: AracSayar Mobile/Models/ and database helper in Data/MeasurementDatabase.cs.

## Deployment Notes

- The FastAPI service is deployable to Render (as currently configured) or any ASGI host; ensure DATABASE_URL and allowed origins are set.
- The detector can run at the edge; point --db-endpoint to the deployed API and keep bandwidth in mind for image payloads.
- Mobile builds target Android, iOS, macOS Catalyst, Windows, and Tizen via .NET MAUI; platform-specific configs live under Platforms/.

## Troubleshooting

- If frames are slow, try a lighter YOLOv8 model (yolov8n.pt) and lower input resolution.
- For tracker instability, switch between deepsort, ioukalman, and sort to match your scene density and motion.
- Backend connection errors: verify DATABASE_URL, open port 8000, and CORS settings; check that /api/detect is reachable from the detector host.

## Contributing

Open issues or PRs are welcome. Please describe the scenario, steps to reproduce, and expected behavior.
