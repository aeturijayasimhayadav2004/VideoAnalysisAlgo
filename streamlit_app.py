"""Streamlit app for real-time-style video analysis.

Features:
- YOLOv5 object detection with adjustable confidence threshold.
- Face recognition against a local `known_faces` directory.
- MediaPipe pose-based activity recognition with aggressive-action alerts.
- Inline alerts for weapons, unknown faces, and aggressive actions.

Run with:
    streamlit run streamlit_app.py

Ensure dependencies are installed (CPU or GPU):
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install opencv-python face_recognition mediapipe streamlit
"""

import json
import tempfile
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import streamlit as st

from video_analysis import (
    AlertManager,
    Detection,
    FaceDatabase,
    FaceMatch,
    PoseActionRecognizer,
    YOLODetector,
    mediapipe,
    optional_import,
    torch,
)

face_recognition = optional_import("face_recognition")


st.set_page_config(page_title="Video Threat & Safety Monitor", layout="wide")

if torch is None:
    st.warning(
        "PyTorch is not installed; YOLO object detection will be disabled until it is available. "
        "Install torch for your platform to enable detections."
    )

if face_recognition is None:
    st.warning("`face_recognition` is not installed; face matching will be disabled until it is available.")

if mediapipe is None:
    st.info(
        "MediaPipe is not installed; pose-based action recognition will be skipped unless you install it."
    )


@st.cache_resource(show_spinner=False)
def load_face_db(root: str, enabled: bool = True) -> FaceDatabase:
    db = FaceDatabase(Path(root), enabled=enabled and face_recognition is not None)
    db.load()
    return db


@st.cache_resource(show_spinner=False)
def load_models(
    conf_threshold: float,
    enable_yolo: bool = True,
    enable_pose: bool = True,
) -> Tuple[YOLODetector, PoseActionRecognizer, AlertManager]:
    detector = YOLODetector(conf_threshold=conf_threshold, enabled=enable_yolo)
    action_recognizer = PoseActionRecognizer(enabled=enable_pose)
    alert_manager = AlertManager()
    return detector, action_recognizer, alert_manager


def draw_annotations(frame: np.ndarray, detections: List[Detection], face_matches: List[FaceMatch], action: str) -> np.ndarray:
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det.label} {det.confidence:.2f}"
        cv2.putText(annotated, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for match in face_matches:
        left, top, right, bottom = match.bbox
        color = (0, 255, 255) if match.name != "unknown" else (0, 0, 255)
        cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)
        cv2.putText(annotated, f"{match.name} ({match.distance:.2f})", (left, top - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(annotated, f"Action: {action}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return annotated


def analyze_video(
    video_path: str,
    detector: YOLODetector,
    face_db: FaceDatabase,
    action_recognizer: PoseActionRecognizer,
    alert_manager: AlertManager,
    conf_threshold: float,
    face_tolerance: float,
    frame_stride: int,
) -> None:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    placeholder = st.empty()
    alert_box = st.empty()
    alerts: List[str] = []
    alert_records: List[dict] = []

    frame_index = 0
    processed = 0
    progress = st.progress(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        if frame_stride > 1 and frame_index % frame_stride != 0:
            continue

        processed += 1
        detections = detector.infer(frame, conf_threshold=conf_threshold)
        matches: List[FaceMatch] = []
        if face_recognition is not None and face_db.enabled:
            rgb_frame = frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_frame)
            encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            for loc, encoding in zip(face_locations, encodings):
                top, right, bottom, left = loc
                name, distance = face_db.match(encoding, tolerance=face_tolerance)
                matches.append(FaceMatch(name=name, bbox=(left, top, right, bottom), distance=distance))

        action = action_recognizer.classify(frame)

        for det in detections:
            alert_manager.check_object(det)
            if det.label.lower() in alert_manager.weapon_labels:
                msg = f"Weapon detected: {det.label} ({det.confidence:.2f})"
                alerts.append(msg)
                alert_records.append({"type": "weapon", "message": msg, "frame": frame_index})
        for match in matches:
            alert_manager.check_face(match)
            if match.name == "unknown":
                msg = f"Unknown face (distance={match.distance:.2f})"
                alerts.append(msg)
                alert_records.append({"type": "unknown_face", "message": msg, "frame": frame_index})
        alert_manager.check_action(action)
        if action in alert_manager.aggressive_actions:
            msg = f"Aggressive action: {action}"
            alerts.append(msg)
            alert_records.append({"type": "aggressive_action", "message": msg, "frame": frame_index})

        annotated = draw_annotations(frame, detections, matches, action)
        placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")

        if alerts:
            alert_box.warning("\n".join(alerts[-5:]))

        progress.progress(min(frame_index / total_frames, 1.0))

    cap.release()
    progress.progress(1.0)
    if not alerts:
        alert_box.info("No threats detected during analysis.")
    else:
        alert_box.info(f"Finished. {len(alerts)} alert(s) recorded.")

    if alert_records:
        st.download_button(
            "Download alert log (JSONL)",
            data="\n".join([json.dumps(rec) for rec in alert_records]),
            file_name="alerts.jsonl",
            mime="application/json",
        )


st.title("Video Threat & Safety Monitor")
st.write("Upload a video to run YOLOv5 object detection, face recognition, and pose-based action analysis.")

with st.expander("How to run this app in your browser", expanded=False):
    st.markdown(
        """
        1. Install dependencies: `pip install torch torchvision torchaudio opencv-python face_recognition mediapipe streamlit`.
        2. Start Streamlit and bind to all interfaces so your browser can connect:  
           `streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true`
        3. Open the URL printed in the terminal (for the command above: http://localhost:8501).
        4. Upload a video and monitor the annotated stream plus alerts in real time.
        """
    )

with st.sidebar:
    st.header("Configuration")
    enable_yolo = st.checkbox("Enable YOLO object detection", value=torch is not None)
    enable_face = st.checkbox("Enable face recognition", value=face_recognition is not None)
    enable_pose = st.checkbox("Enable pose/action recognition", value=mediapipe is not None)
    conf_threshold = st.slider("Object confidence threshold", 0.1, 0.9, 0.35, 0.05)
    face_tolerance = st.slider("Face match tolerance", 0.3, 0.8, 0.5, 0.05)
    frame_stride = st.slider("Process every Nth frame", 1, 5, 1, 1)
    st.markdown("Known faces folder: `known_faces/<person>/image.jpg`")

uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    face_db = load_face_db("known_faces", enabled=enable_face)
    detector, action_recognizer, alert_manager = load_models(
        conf_threshold, enable_yolo=enable_yolo, enable_pose=enable_pose
    )
    if not enable_face:
        st.info("Face matching disabled. Enable it and install `face_recognition` for recognition alerts.")
    if not enable_yolo:
        st.info("YOLO object detection disabled. Install torch to enable detections.")
    if not enable_pose:
        st.info("Pose/action recognition disabled. Install MediaPipe to enable it.")
    st.success("Models ready. Starting analysis...")
    analyze_video(
        video_path,
        detector,
        face_db,
        action_recognizer,
        alert_manager,
        conf_threshold,
        face_tolerance,
        frame_stride,
    )
else:
    st.info("Upload a video to begin analysis. Ensure the server has access to `known_faces` for recognition.")

