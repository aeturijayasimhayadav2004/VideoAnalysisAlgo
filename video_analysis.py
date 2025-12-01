"""
Real-time video analysis script combining object detection (YOLOv5),
face recognition, pose-based activity recognition, and alerting.

Usage examples:
    python video_analysis.py --source 0
    python video_analysis.py --source path/to/video.mp4 --no-display

Dependencies (install as needed):
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install opencv-python face_recognition mediapipe
    # YOLOv5 weights are downloaded automatically via torch.hub on first run.

The script expects a `known_faces` directory with subfolders named after people.
Each subfolder should contain one or more reference images for that person.
"""

import argparse
import importlib
import importlib.util
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


def optional_import(module_name: str):
    """Return a module if available, otherwise None (no try/except around imports)."""

    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return None
    return importlib.import_module(module_name)


face_recognition = optional_import("face_recognition")
mediapipe = optional_import("mediapipe")
torch = optional_import("torch")
mp = mediapipe


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]


@dataclass
class FaceMatch:
    name: str
    bbox: Tuple[int, int, int, int]
    distance: float


class FaceDatabase:
    """Load and store known face encodings from a directory tree."""

    def __init__(self, root: Path, enabled: bool = True):
        self.root = root
        self.encodings: List[np.ndarray] = []
        self.labels: List[str] = []
        self.enabled = enabled and face_recognition is not None

    def load(self) -> None:
        if not self.enabled:
            print("[FaceDatabase] face_recognition not available; skipping known face loading.")
            return
        if not self.root.exists():
            print(f"[FaceDatabase] No known faces directory found at {self.root.resolve()}.")
            return

        for person_dir in self.root.iterdir():
            if not person_dir.is_dir():
                continue
            for image_path in person_dir.glob("*.*"):
                image = face_recognition.load_image_file(str(image_path))
                face_locations = face_recognition.face_locations(image)
                if not face_locations:
                    continue
                encoding = face_recognition.face_encodings(image, face_locations)[0]
                self.encodings.append(encoding)
                self.labels.append(person_dir.name)
                print(f"[FaceDatabase] Loaded face for '{person_dir.name}' from {image_path.name}.")

    def match(self, face_encoding: np.ndarray, tolerance: float = 0.5) -> Tuple[str, float]:
        if not self.enabled:
            return "unknown", 1.0
        if not self.encodings:
            return "unknown", 1.0
        distances = face_recognition.face_distance(self.encodings, face_encoding)
        best_index = int(np.argmin(distances))
        best_distance = distances[best_index]
        if best_distance <= tolerance:
            return self.labels[best_index], float(best_distance)
        return "unknown", float(best_distance)


class PoseActionRecognizer:
    """Heuristic action recognizer based on MediaPipe pose landmarks."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and mediapipe is not None
        self.pose = None
        if self.enabled:
            self.pose = mediapipe.solutions.pose.Pose(static_image_mode=False, model_complexity=1)
        else:
            print("[PoseActionRecognizer] MediaPipe not available; pose-based actions disabled.")
        self.previous_landmarks: Optional[np.ndarray] = None
        self.last_action: str = "idle" if self.enabled else "pose_unavailable"
        self.cooldown = 5  # frames to stabilize predictions
        self.counter = 0

    def _landmark_array(self, results) -> Optional[np.ndarray]:
        if not self.enabled or results is None or results.pose_landmarks is None:
            return None
        if not results.pose_landmarks:
            return None
        coords = []
        for lm in results.pose_landmarks.landmark:
            coords.append([lm.x, lm.y, lm.z])
        return np.array(coords)

    def classify(self, frame: np.ndarray) -> str:
        if not self.enabled or self.pose is None:
            return self.last_action
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        landmarks = self._landmark_array(results)
        if landmarks is None:
            self.last_action = "idle"
            return self.last_action

        action = "walking"
        # Simple heuristic: overall speed to differentiate walking vs running.
        if self.previous_landmarks is not None:
            displacement = np.linalg.norm(landmarks[:, :2] - self.previous_landmarks[:, :2], axis=1)
            speed = float(np.mean(displacement))
            if speed > 0.03:
                action = "running"
        # Aggressive heuristic: large arm extension difference could indicate fighting.
        left_wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value]
        left_elbow = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value]
        arm_span = np.linalg.norm(left_wrist[:2] - right_wrist[:2])
        elbow_span = np.linalg.norm(left_elbow[:2] - right_elbow[:2])
        if arm_span > 0.6 and elbow_span > 0.4:
            action = "fighting"

        # Stabilize predictions over a few frames to reduce flicker.
        if action != self.last_action:
            self.counter += 1
            if self.counter >= self.cooldown:
                self.last_action = action
                self.counter = 0
        else:
            self.counter = 0
            self.last_action = action

        self.previous_landmarks = landmarks
        return self.last_action


class AlertManager:
    def __init__(
        self,
        weapon_labels: Optional[List[str]] = None,
        aggressive_actions: Optional[List[str]] = None,
        log_path: Optional[Path] = None,
    ):
        self.weapon_labels = weapon_labels or ["knife", "gun", "pistol", "rifle"]
        self.aggressive_actions = aggressive_actions or ["fighting", "aggressive"]
        self.log_path = log_path
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("", encoding="utf-8")

    def check_object(self, detection: Detection) -> None:
        if detection.label.lower() in self.weapon_labels:
            message = f"[ALERT] Weapon detected: {detection.label} ({detection.confidence:.2f})"
            print(message)
            self._log_event("weapon", message)

    def check_face(self, face_match: FaceMatch) -> None:
        if face_match.name == "unknown":
            message = f"[ALERT] Unknown face detected (distance={face_match.distance:.2f})."
            print(message)
            self._log_event("unknown_face", message)

    def check_action(self, action: str) -> None:
        if action in self.aggressive_actions:
            message = f"[ALERT] Aggressive action detected: {action}."
            print(message)
            self._log_event("aggressive_action", message)

    def _log_event(self, event_type: str, message: str) -> None:
        if not self.log_path:
            return
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{event_type}: {message}\n")


class YOLODetector:
    def __init__(
        self,
        device: Optional[str] = None,
        model_name: str = "yolov5s",
        conf_threshold: float = 0.35,
        enabled: bool = True,
    ):
        self.available = enabled and torch is not None
        self.device = "cpu"
        self.conf_threshold = conf_threshold
        self.model = None
        if not enabled:
            print("[YOLO] Detector explicitly disabled by configuration.")
            return
        if torch is None:
            print("[YOLO] Torch not available; YOLO detector disabled.")
            return
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model = torch.hub.load("ultralytics/yolov5", model_name, pretrained=True)
            self.model.to(self.device)
            self.model.eval()
            self.model.conf = conf_threshold
            print(f"[YOLO] Loaded {model_name} on {self.device} (conf>{self.conf_threshold})")
        except Exception as exc:  # noqa: BLE001
            print(f"[YOLO] Failed to load model: {exc}")
            self.available = False
            self.model = None

    def infer(self, frame: np.ndarray, conf_threshold: Optional[float] = None) -> List[Detection]:
        if not self.available or self.model is None:
            return []
        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold
        results = self.model(frame)
        detections: List[Detection] = []
        for *xyxy, conf, cls in results.xyxy[0].tolist():
            if float(conf) < threshold:
                continue
            x1, y1, x2, y2 = map(int, xyxy)
            label = self.model.names[int(cls)]
            detections.append(Detection(label=label, confidence=float(conf), bbox=(x1, y1, x2, y2)))
        return detections


class AnalyzerThread(threading.Thread):
    def __init__(self, frame_queue: queue.Queue, result_queue: queue.Queue, face_db: FaceDatabase, detector: YOLODetector, alert_manager: AlertManager, action_recognizer: PoseActionRecognizer):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.face_db = face_db
        self.detector = detector
        self.alert_manager = alert_manager
        self.action_recognizer = action_recognizer
        self.running = True

    def run(self) -> None:
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            detections = self.detector.infer(frame)
            face_matches = self._recognize_faces(frame)
            action = self.action_recognizer.classify(frame)

            for det in detections:
                self.alert_manager.check_object(det)
            for match in face_matches:
                self.alert_manager.check_face(match)
            self.alert_manager.check_action(action)

            self.result_queue.put((frame, detections, face_matches, action))

    def _recognize_faces(self, frame: np.ndarray) -> List[FaceMatch]:
        if face_recognition is None or not self.face_db.enabled:
            return []
        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        matches: List[FaceMatch] = []
        for loc, encoding in zip(face_locations, encodings):
            top, right, bottom, left = loc
            name, distance = self.face_db.match(encoding)
            matches.append(FaceMatch(name=name, bbox=(left, top, right, bottom), distance=distance))
        return matches


class RealTimeVideoAnalyzer:
    def __init__(
        self,
        source: str,
        display: bool,
        max_queue: int = 3,
        enable_face: bool = True,
        enable_pose: bool = True,
        alert_log: Optional[str] = None,
    ):
        self.source = source
        self.display = display
        self.frame_queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self.result_queue: queue.Queue = queue.Queue(maxsize=max_queue)

        self.face_db = FaceDatabase(Path("known_faces"), enabled=enable_face)
        self.face_db.load()

        self.yolo_detector = YOLODetector()
        log_path = Path(alert_log) if alert_log else None
        self.alert_manager = AlertManager(log_path=log_path)
        self.action_recognizer = PoseActionRecognizer(enabled=enable_pose)

        if not self.yolo_detector.available:
            print("[System] YOLO detector unavailable; object detection will be skipped.")

        self.analyzer_thread = AnalyzerThread(
            self.frame_queue,
            self.result_queue,
            self.face_db,
            self.yolo_detector,
            self.alert_manager,
            self.action_recognizer,
        )

    def start(self) -> None:
        cap = cv2.VideoCapture(0 if self.source == "0" else self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")

        self.analyzer_thread.start()
        print("[System] Press 'q' to exit.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())

                self._process_results()

                if self.display:
                    cv2.imshow("Real-Time Analysis", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            self._shutdown(cap)

    def _process_results(self) -> None:
        try:
            while True:
                frame, detections, face_matches, action = self.result_queue.get_nowait()
                self._draw_annotations(frame, detections, face_matches, action)
        except queue.Empty:
            pass

    def _draw_annotations(self, frame: np.ndarray, detections: List[Detection], face_matches: List[FaceMatch], action: str) -> None:
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det.label} {det.confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for match in face_matches:
            left, top, right, bottom = match.bbox
            color = (0, 255, 255) if match.name != "unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, f"{match.name} ({match.distance:.2f})", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(frame, f"Action: {action}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    def _shutdown(self, cap: cv2.VideoCapture) -> None:
        self.analyzer_thread.running = False
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time video analysis with YOLOv5, face recognition, and pose estimation.")
    parser.add_argument("--source", default="0", help="Video source (0 for webcam or path to video file).")
    parser.add_argument("--no-display", action="store_true", dest="no_display", help="Disable frame display window.")
    parser.add_argument("--max-queue", type=int, default=3, help="Maximum queued frames for analysis thread.")
    parser.add_argument("--disable-face", action="store_true", help="Disable face recognition even if installed.")
    parser.add_argument("--disable-pose", action="store_true", help="Disable pose-based activity recognition.")
    parser.add_argument(
        "--alert-log",
        type=str,
        default=None,
        help="Optional path to append alert events (weapons, unknown faces, aggressive actions).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyzer = RealTimeVideoAnalyzer(
        source=args.source,
        display=not args.no_display,
        max_queue=args.max_queue,
        enable_face=not args.disable_face,
        enable_pose=not args.disable_pose,
        alert_log=args.alert_log,
    )
    analyzer.start()


if __name__ == "__main__":
    main()
