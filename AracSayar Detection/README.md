# Vehicle Counting Pipeline

Real-time vehicle detection, tracking, and counting pipeline built around YOLOv8 with interchangeable trackers (DeepSORT, IOU+Kalman, SORT). The system handles camera or video input, overlays counts, and can post events to a backend.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r deep_sort/requirements.txt  # plus ultralytics for YOLO
python main.py --source testVideo.mp4 --display --tracker deepsort
```

Key flags:
- `--source`: camera index or video file path
- `--tracker`: `deepsort` (default), `ioukalman`, or `sort`
- `--count-mode`: `single` or `double` line counting
- `--orientation`: `auto`, `horizontal`, or `vertical`
- `--db-endpoint`: HTTP endpoint to receive vehicle events

## Project Layout

- `main.py`: CLI entrypoint; wires detector, tracker, counting options, logging, and DB client.
- `src/pipeline.py`: Core pipeline loop (frame grab, detection, tracking, counting, visualization, DB event posting). Contains `CountingController` for line management and count logic.
- `src/__init__.py`: Package marker.

### Detectors
- `src/detectors/yolo.py`: YOLOv8 wrapper filtering vehicle classes and returning normalized `Detection` objects.
- `src/detectors/__init__.py`: Exports detector types.

### Tracking
- `src/tracking/deep_sort_adapter.py`: Adapter around the bundled `deep_sort` repo; converts detections and returns `TrackState` objects.
- `src/tracking/iou_module.py`: IOU + Kalman filter tracker with basic data association.
- `src/tracking/sort_tracker.py`: Simplified SORT-style tracker.
- `src/tracking/__init__.py`: Exports tracker classes.

### Utilities
- `src/utils/types.py`: Dataclasses for detections and track states.
- `src/utils/drawing.py`: Overlay helper for tracks, lines, and per-class counts.
- `src/utils/fps_logger.py`: Lightweight FPS meter.
- `src/utils/road.py`: Estimates motion band and orientation from background subtraction.
- `src/utils/log_screen.py`: Optional side log window renderer.
- `src/utils/db_client.py`: HTTP client for posting vehicle events (base64-encoded crops) to `DEFAULT_DB_ENDPOINT`.
- `src/utils/__init__.py`: Utility package marker.

### Third-Party Bundles
- `deep_sort/`: Local copy of DeepSORT library used by the default tracker. Install its requirements when using `--tracker deepsort`.

### Data & Outputs
- `log/pipeline.log`: Runtime log output (created automatically).
- `yolov8n.pt`: Default YOLOv8 model weights (replaceable).
- `testVideo*.mp4`: Sample videos (ignored by git).

## Notes

- Requires Python 3.10+ and OpenCV with video support.
- If DB posting is enabled, ensure `requests` is installed and the endpoint is reachable.
- Press `q` in the display window to stop the pipeline.
