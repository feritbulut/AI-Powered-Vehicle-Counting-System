
---

# AI-Based Vehicle Detection & Monitoring System - Backend API

## 🌍 Live System Status (Deployment)

The backend infrastructure is currently **deployed and active**. You can access the live system and documentation using the links below:

* **Base URL (Render):** `https://ai-vehicle-detection-backend-api.onrender.com`
* **Interactive API Documentation (Swagger UI):** [Click to View / Test API](https://www.google.com/search?q=https://ai-vehicle-detection-backend-api.onrender.com/docs)
* **Database Host:** Aiven Cloud (MySQL)
* **WebSocket Stream:** `wss://ai-vehicle-detection-backend-api.onrender.com/ws`

---

## 📖 Project Description

This repository hosts the **Backend API infrastructure** for a real-time vehicle detection system. The project is designed to bridge the gap between an **AI-powered computer vision unit** and a **mobile monitoring application**.

The system processes vehicle data (images, types, tracking IDs, and timestamps) detected by edge devices and synchronizes this information with mobile clients instantly using **WebSocket** technology, while simultaneously logging historical data into a **MySQL database**.

## 🏗️ System Architecture & Workflow

The backend acts as the central nervous system of the project, managing data flow in two directions:

1. **Ingestion (REST API):** The AI/Camera unit sends HTTP POST requests containing vehicle metadata and Base64-encoded images to the server.
2. **Broadcasting (WebSocket):** The server processes this data and pushes real-time notifications to connected mobile applications without requiring page refreshes.
3. **Persistence (MySQL):** All traffic data is structured and stored for future reporting and statistical analysis using SQLAlchemy models.

## 🚀 Key Features

* **Hybrid Communication Protocol:** Implements both **REST API** (for AI data ingestion) and **WebSocket** (for real-time client updates).
* **Real-Time Image Transmission:** Handles vehicle snapshots (Base64/URL strings) to display visual data on mobile devices instantly.
* **Persistent Logging:** Detailed logging of vehicle types, unique tracking IDs, detection timestamps, and camera sources using **SQLAlchemy ORM**.
* **Dynamic Statistics:** Provides endpoints for fetching total vehicle counts and recent activity logs.

## 🛠️ Tech Stack

* **Framework:** FastAPI (Python)
* **Database:** MySQL / SQLAlchemy (Hosted on Aiven)
* **Protocol:** HTTP (REST) & WebSocket (WSS)
* **Data Validation:** Pydantic
* **Server:** Uvicorn (ASGI) / Render Cloud

## 🔌 API Endpoints Integration

| Method | Endpoint | Description | Integration Note |
| --- | --- | --- | --- |
| `POST` | `/api/detect` | Receives vehicle data (ID, Type, Image) from the AI unit. | Used by **AI Team** to push data. |
| `WS` | `/ws` | WebSocket channel for real-time mobile app communication. | Used by **Mobile Team** to listen for updates. |
| `GET` | `/api/stats` | Returns total vehicle count and recent logs. | Used for dashboard statistics. |
| `DELETE` | `/api/delete/{id}` | Removes an erroneous entry from the database. | Admin/Cleanup usage. |

---

### 🧪 How to Test

1. Navigate to the **[Swagger UI](https://www.google.com/search?q=https://ai-vehicle-detection-backend-api.onrender.com/docs)**.
2. Use the `POST /api/detect` endpoint to simulate a vehicle detection.
3. Connect to the WebSocket (`/ws`) via a client (e.g., Postman) to observe the real-time broadcast of the data you just posted.