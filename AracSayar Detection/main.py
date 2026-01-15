import argparse
import logging
from pathlib import Path

from src.detectors import YOLODetector
from src.pipeline import Pipeline
from src.tracking import DeepSortTracker, IOUKalmanTracker, SortTracker
from src.utils.db_client import DEFAULT_DB_ENDPOINT, DatabaseClient


def parse_args():
    parser = argparse.ArgumentParser(
        description="Vehicle detection pipeline (YOLO + tracker with optional fallbacks)"
    )
    parser.add_argument(
        "--source", default="0", help="Camera index or video file path (default: 0)"
    )
    parser.add_argument(
        "--display",
        action="store_true",
        default=False,
        help="Show video window (off by default for headless devices).",
    )
    parser.add_argument(
        "--tracker",
        choices=["ioukalman", "sort", "deepsort"],
        default="deepsort",
        help="Tracker type: DeepSORT (default), IOU+Kalman, or SORT.",
    )
    parser.add_argument(
        "--yolo-model",
        default="yolov8n.pt",
        help="Path or name of YOLO model (used when --detector yolo).",
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=0.25,
        help="Confidence threshold for YOLO detector.",
    )
    parser.add_argument(
        "--yolo-device",
        default=None,
        help="Device for YOLO (e.g., 'cpu', '0'); leave empty for auto.",
    )
    parser.add_argument(
        "--count-mode",
        choices=["single", "double"],
        default="double",
        help="Counting style: single line (exit-only) or double line (entry/exit band).",
    )
    parser.add_argument(
        "--orientation",
        choices=["auto", "horizontal", "vertical"],
        default="auto",
        help="Manually force horizontal/vertical flow axis or auto-detect.",
    )
    parser.add_argument(
        "--single-line-pos",
        type=int,
        default=None,
        help="Override single-line position (pixels along movement axis).",
    )
    parser.add_argument(
        "--start-line-pos",
        type=int,
        default=None,
        help="Override double-line start position (pixels along movement axis).",
    )
    parser.add_argument(
        "--end-line-pos",
        type=int,
        default=None,
        help="Override double-line end position (pixels along movement axis).",
    )
    parser.add_argument(
        "--line-offset",
        type=float,
        default=0.15,
        help="Percent (0-1) offset before exit edge for auto line placement.",
    )
    parser.add_argument(
        "--line-gap",
        type=float,
        default=0.2,
        help="Percent (0-1) gap between start/end in double mode when auto placing.",
    )
    parser.add_argument(
        "--db-endpoint",
        default=None,
        help="HTTP endpoint to POST vehicle events (JSON). Defaults to the hosted API.",
    )
    parser.add_argument(
        "--db-timeout",
        type=float,
        default=5.0,
        help="Timeout for DB POST requests in seconds.",
    )
    return parser.parse_args()


def configure_logging():
    log_dir = Path(__file__).resolve().parent / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pipeline.log"

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # clear previous handlers to avoid duplicates
    root.handlers = []
    root.addHandler(file_handler)
    root.addHandler(stream_handler)


def main():
    args = parse_args()
    configure_logging()

    display = args.display

    try:
        detector = YOLODetector(
            model_path=args.yolo_model, conf=args.yolo_conf, device=args.yolo_device
        )
        logging.info("Using YOLO detector with model %s", args.yolo_model)
    except ImportError as exc:
        logging.error(
            "ultralytics/YOLO unavailable (%s). Please install it to run the pipeline.",
            exc,
        )
        raise SystemExit(1) from exc

    if args.tracker == "sort":
        tracker = SortTracker()
    elif args.tracker == "deepsort":
        try:
            tracker = DeepSortTracker()
            logging.info("Using DeepSORT tracker.")
        except Exception as exc:
            logging.warning(
                "DeepSORT unavailable (%s); falling back to IOU+Kalman tracker.", exc
            )
            tracker = IOUKalmanTracker()
    else:
        tracker = IOUKalmanTracker()
        logging.info("Using IOU+Kalman tracker.")

    db_endpoint = args.db_endpoint or DEFAULT_DB_ENDPOINT
    db_client = DatabaseClient(db_endpoint, timeout=args.db_timeout)

    forced_orientation = None if args.orientation == "auto" else args.orientation

    pipeline = Pipeline(
        source=args.source,
        detector=detector,
        tracker=tracker,
        display=display,
        count_mode=args.count_mode,
        single_line_pos=args.single_line_pos if args.count_mode == "single" else None,
        start_line_pos=args.start_line_pos if args.count_mode == "double" else None,
        end_line_pos=args.end_line_pos if args.count_mode == "double" else None,
        line_offset_pct=args.line_offset,
        line_gap_pct=args.line_gap,
        forced_orientation=forced_orientation,
        db_client=db_client,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
