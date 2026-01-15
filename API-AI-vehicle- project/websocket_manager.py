from fastapi import WebSocket
from typing import List

class ConnectionManager:
    """
    Manages active WebSocket connections for real-time updates.
    """
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        """
        Sends a message to all currently connected clients.
        Includes error handling to remove stale connections.
        """
        # Iterate over a copy of the list to allow safe modification during iteration
        for connection in self.active_connections[:]:
            try:
                await connection.send_text(message)
            except:
                # If sending fails, assume client disconnected
                self.disconnect(connection)

manager = ConnectionManager()