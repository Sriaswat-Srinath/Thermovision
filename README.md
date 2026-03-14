# 🛰️ ThermoVision: Multispectral Fusion Detection

**ThermoVision** is a state-of-the-art multispectral object detection system that fuses RGB and Infrared (IR) data using YOLOv8 and advanced computer vision techniques. It provides real-time detection, distance estimation, and mask fusion with a high-performance FastAPI backend.

---

## ✨ Key Features

- **🚀 Dual-Stream Fusion**: Intelligently merges detections from RGB and Infrared cameras to improve reliability in low-light or obscured environments.
- **🌡️ IR Simulation**: Built-in CLAHE-based IR simulation allows testing the fusion logic using standard RGB input when an IR camera is unavailable.
- **📏 Distance Estimation**: Real-time distance calculation for detected objects using focal length and class-specific real-height benchmarks.
- **🎭 Mask Fusion & Segmentation**: Advanced weighted mask fusion for precise object segmentation.
- **🎥 Video Batch Processing**: Asynchronous background jobs for processing full-length videos with status polling and download capabilities.
- **📊 Interactive UI**: A futuristic, high-performance dashboard for live visualization and batch uploads.

---

## 🛠️ Tech Stack

- **Backend**: Python 3.11+, FastAPI, Uvicorn
- **AI/ML**: YOLOv8 (ultralytics), PyTorch, Supervision
- **Vision**: OpenCV, NumPy
- **Containerization**: Docker, Docker Compose

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11 or higher
- (Recommended) NVIDIA GPU with CUDA support

### Local Installation
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Thermovision
   ```

2. **Setup Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the API**:
   ```bash
   uvicorn app.main:app --reload
   ```

---

## 🐳 Docker Deployment

ThermoVision is fully containerized for easy deployment:

```bash
docker-compose up --build
```
*The default configuration includes NVIDIA GPU support. To run on CPU, remove the `deploy` section in `docker-compose.yml`.*

---

## 📡 API Overview

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/health` | `GET` | System status and model health. |
| `/detect/image` | `POST` | Process RGB + IR images for fused detections. |
| `/detect/image/visualize` | `POST` | Returns a side-by-side annotated visualization. |
| `/video/process` | `POST` | Queue a video file for batch fusion processing. |
| `/video/status/{job_id}` | `GET` | Poll the status of a video job. |

---

## 📂 Project Structure

```text
Thermovision/
├── app/                  # Main application logic
├── static/               # Web-based dashboard
├── tests/                # Unit and integration tests
├── models/               # YOLOv8 weight storage (ignored by Git)
├── uploads/              # Temporary job storage
└── outputs/              # Processed result storage
```

---

## 📜 License
This project is for academic/research purposes. (Add your license here).
