# Video Analysis Toolkit

This project provides two entry points for real-time and offline video threat analysis:

1. **`video_analysis.py`** – threaded OpenCV pipeline for webcam or file sources with YOLOv5 object detection, face recognition, MediaPipe pose analysis, and alerting.
2. **`streamlit_app.py`** – lightweight Streamlit UI for uploading a video, tuning thresholds, and visualizing annotated frames and alerts in the browser.

## Setup

Install dependencies (GPU-enabled PyTorch optional):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python face_recognition mediapipe streamlit
```

If a dependency is unavailable (e.g., `face_recognition` or `mediapipe` fails to install), the tools will continue running with
the affected feature disabled and print a warning so you can still exercise the rest of the pipeline.

Prepare a face database:

```
known_faces/
  Alice/
    alice1.jpg
  Bob/
    bob.png
```

## Running the threaded analyzer

```bash
python video_analysis.py --source 0           # webcam
python video_analysis.py --source video.mp4   # file
python video_analysis.py --source 0 --disable-face   # run without face recognition
python video_analysis.py --source 0 --disable-pose   # run without pose analysis
python video_analysis.py --source rtsp://... --alert-log logs/alerts.txt   # remote stream + alert logging
```

Press `q` to exit when the display window is enabled.

## Running the Streamlit app (in your browser)

Launch the app and bind it to all interfaces so you can open it from your browser (or a remote workstation when running in a container or VM):

```bash
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
```

Open the printed URL (for the example above, `http://localhost:8501`) in your browser, upload a video, and watch annotated frames and alerts stream inline. Use the sidebar to adjust confidence thresholds, face tolerance, and frame stride for faster processing. Toggle object detection, face recognition, or pose-based action recognition off if a dependency is missing or you want to benchmark individual components.

## Capability coverage

- **Video input**: accepts webcam (`--source 0`), local files, and remote streams (e.g., RTSP URLs) via OpenCV’s `VideoCapture`.
- **Feature extraction**: YOLOv5 detects objects (including weapons), `face_recognition` encodes faces, and MediaPipe Pose extracts landmarks for action heuristics.
- **Model usage**: uses pretrained YOLOv5 weights and your provided face images under `known_faces/`; no custom training is bundled, but you can swap models or add new weights without code changes.
- **Analysis & interpretation**: the analyzer and Streamlit app run per-frame detection, face matching, and pose-based action recognition to flag threats in real time.
- **Outputs**: on-screen overlays, console alerts, optional alert log file (`--alert-log`), and a downloadable JSONL alert log from the Streamlit app for reporting or downstream workflows.

