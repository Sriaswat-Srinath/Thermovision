"""
ThermoVision — FastAPI Backend
RGB + Infrared Fusion Detection using YOLOv8
"""

import os, io, uuid, time
import numpy as np
import cv2
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from ultralytics import YOLO
from torchvision.ops import nms as torch_nms
import supervision as sv

# ── Fix for PyTorch 2.6+ weights_only=True default ────────────────────
# Allowlist all classes needed to deserialize YOLOv8 .pt files
try:
    safe = [
        torch.nn.modules.container.Sequential,
        torch.nn.modules.container.ModuleList,
        torch.nn.modules.container.ModuleDict,
        torch.nn.modules.batchnorm.BatchNorm2d,
        torch.nn.modules.conv.Conv2d,
        torch.nn.modules.activation.SiLU,
        torch.nn.modules.activation.ReLU,
        torch.nn.modules.pooling.MaxPool2d,
        torch.nn.modules.upsampling.Upsample,
        torch.nn.modules.linear.Linear,
    ]
    # Add ultralytics-specific classes
    try:
        from ultralytics.nn.tasks import DetectionModel, SegmentationModel
        from ultralytics.nn.modules import (
            Conv, C2f, SPPF, Detect, DFL, C3, Bottleneck,
        )
        safe += [DetectionModel, SegmentationModel, Conv, C2f, SPPF, Detect, DFL, C3, Bottleneck]
    except ImportError:
        pass
    torch.serialization.add_safe_globals(safe)
except AttributeError:
    pass  # PyTorch < 2.6, no action needed

# ── App ────────────────────────────────────────────────────────────────
app = FastAPI(title="ThermoVision API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── Config ─────────────────────────────────────────────────────────────
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
RGB_MODEL_PATH = os.getenv("RGB_MODEL_PATH", "models/yolov8m-seg.pt")
IR_MODEL_PATH  = os.getenv("IR_MODEL_PATH",  "models/ir_yolov8m.pt")
CONF_THRESH    = float(os.getenv("CONF_THRESH", "0.15")) # Lower default for WBF/Union
NMS_IOU        = float(os.getenv("NMS_IOU",     "0.50"))

class FusionConfig:
    conf = CONF_THRESH
    # ---- Adaptive Confidence ----
    class_conf = {
        "person":     0.25,
        "car":        0.20,
        "truck":      0.20,
        "bus":        0.20,
        "motorcycle": 0.30,
        "bicycle":    0.30,
    }
    # ---- Distance Estimation ----
    focal_length_px = 1000.0
    distance_uncertainty_pct = 0.15
    ema_alpha = 0.30
    real_heights = {
        "person":     1.75,
        "car":        1.5,
        "truck":      3.0,
        "bus":        3.2,
        "motorcycle": 1.2,
        "bicycle":    1.2,
    }
    # ---- Mask Fusion ----
    mask_fusion_method = "weighted"
    mask_rgb_weight = 0.60
    mask_thermal_weight = 0.40

CFG = FusionConfig()

Path("uploads").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

# ── IR label → COCO name mapping ──────────────────────────────────────
# Update these to match your actual IR model class names
IR_TO_COCO = {
    "person":  "person",
    "bike":    "bicycle",
    "motor":   "motorcycle",
    "car":     "car",
    "bus":     "bus",
    "truck":   "truck",
    "light":   "traffic light",
    "hydrant": "fire hydrant",
    "sign":    "stop sign",
}

# ── Lazy model loader ──────────────────────────────────────────────────
_rgb_model = None
_ir_model  = None

def _load_yolo(path: str):
    """Load YOLO model with PyTorch 2.6 weights_only=False fix."""
    _orig = torch.load
    torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
    try:
        model = YOLO(path)
    finally:
        torch.load = _orig   # always restore
    return model

def get_models():
    global _rgb_model, _ir_model
    if _rgb_model is None:
        print(f"[LOAD] RGB model → {RGB_MODEL_PATH}  ({DEVICE})")
        try:
            _rgb_model = _load_yolo(RGB_MODEL_PATH)
            _rgb_model.to(DEVICE)
        except Exception as e:
            print(f"[ERROR] {e}")
            raise RuntimeError(f"Cannot load RGB model: {e}")
    if _ir_model is None:
        print(f"[LOAD] IR  model → {IR_MODEL_PATH}  ({DEVICE})")
        try:
            _ir_model = _load_yolo(IR_MODEL_PATH)
            _ir_model.to(DEVICE)
        except Exception as e:
            print(f"[ERROR] {e}")
            raise RuntimeError(f"Cannot load IR model: {e}")
    return _rgb_model, _ir_model

# ── Schemas ────────────────────────────────────────────────────────────
class Box(BaseModel):
    label: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float
    source: str
    polygon: Optional[list[list[float]]] = None
    track_id: Optional[int] = None
    distance: Optional[float] = None

class FusionResult(BaseModel):
    rgb_detections: list[Box]
    ir_detections: list[Box]
    fusion_detections: list[Box]
    rgb_count: int
    ir_count: int
    fusion_count: int
    avg_conf_rgb: float
    avg_conf_ir: float
    avg_conf_fusion: float
    processing_ms: float
    device: str

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    frames_done:  int
    total_frames: int
    output_path: Optional[str] = None
    error:       Optional[str] = None

# ── Video jobs ─────────────────────────────────────────────────────────
video_jobs: dict[str, dict] = {}

# ── Global state ───────────────────────────────────────────────────────
tracker = sv.ByteTrack(
    track_activation_threshold=0.2,
    lost_track_buffer=60,
    minimum_matching_threshold=0.3,
    frame_rate=20,
    minimum_consecutive_frames=2
)
distance_ema: dict[int, float] = {}

@app.post("/tracker/reset")
def reset_tracker():
    """Reset tracker state for new video streams."""
    global tracker, distance_ema
    tracker = sv.ByteTrack(
        track_activation_threshold=0.2,
        lost_track_buffer=60,
        minimum_matching_threshold=0.3,
        frame_rate=20,
        minimum_consecutive_frames=2
    )
    distance_ema.clear()
    return {"status": "ok"}

# ── Helpers ────────────────────────────────────────────────────────────
def iou(boxA, boxB) -> float:
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter  = max(0, xB - xA) * max(0, yB - yA)
    areaA  = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB  = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union  = areaA + areaB - inter
    return inter / union if union > 0 else 0.0

def fuse_detections_union(rgb_boxes, rgb_scores, rgb_classes, th_boxes, th_scores, th_classes, iou_thresh: float = 0.50):
    if len(rgb_boxes) == 0 and len(th_boxes) == 0:
        return np.empty((0,4)), np.array([]), np.array([], dtype=int), np.array([], dtype=int)

    all_boxes, all_scores, all_classes, all_source = [], [], [], []

    for b, s, c in zip(rgb_boxes, rgb_scores, rgb_classes):
        all_boxes.append(b); all_scores.append(s); all_classes.append(c); all_source.append(0)
    for b, s, c in zip(th_boxes, th_scores, th_classes):
        all_boxes.append(b); all_scores.append(s); all_classes.append(c); all_source.append(1)

    if not all_boxes:
        return np.empty((0,4)), np.array([]), np.array([], dtype=int), np.array([], dtype=int)

    all_boxes   = np.array(all_boxes,   dtype=np.float32)
    all_scores  = np.array(all_scores,  dtype=np.float32)
    all_classes = np.array(all_classes, dtype=int)
    all_source  = np.array(all_source,  dtype=int)

    # Sort descending by conf
    order = np.argsort(-all_scores)
    all_boxes, all_scores, all_classes, all_source = all_boxes[order], all_scores[order], all_classes[order], all_source[order]

    kept_boxes, kept_scores, kept_classes, kept_source = [], [], [], []

    for i in range(len(all_boxes)):
        duplicate = False
        for j, kb in enumerate(kept_boxes):
            if all_classes[i] == kept_classes[j] and iou(all_boxes[i], kb) > iou_thresh:
                if all_source[i] != kept_source[j]:
                    kept_source[j] = 2
                duplicate = True
                break
        if not duplicate:
            kept_boxes.append(all_boxes[i])
            kept_scores.append(all_scores[i])
            kept_classes.append(all_classes[i])
            kept_source.append(int(all_source[i]))

    return np.array(kept_boxes), np.array(kept_scores), np.array(kept_classes, dtype=int), np.array(kept_source, dtype=int)

def apply_class_conf_filter(boxes, scores, classes, names):
    keep = []
    for i, (c, s) in enumerate(zip(classes, scores)):
        label = names[c]
        if s >= CFG.class_conf.get(label, CFG.conf):
            keep.append(i)
    if not keep: return np.empty((0, 4)), np.array([]), np.array([], dtype=int)
    keep = np.array(keep)
    return boxes[keep], scores[keep], classes[keep]

def estimate_distance(label: str, pixel_height: float):
    if label not in CFG.real_heights or pixel_height <= 0:
        return None, None
    d = (CFG.real_heights[label] * CFG.focal_length_px) / pixel_height
    sigma = d * CFG.distance_uncertainty_pct
    return d, sigma

def smooth_distance(track_id: int, new_dist: float) -> float:
    if track_id not in distance_ema:
        distance_ema[track_id] = new_dist
    else:
        distance_ema[track_id] = CFG.ema_alpha * new_dist + (1 - CFG.ema_alpha) * distance_ema[track_id]
    return distance_ema[track_id]

def fuse_masks(rgb_mask: np.ndarray, thermal_mask: np.ndarray) -> np.ndarray:
    if CFG.mask_fusion_method == "union":
        return np.logical_or(rgb_mask > 0.5, thermal_mask > 0.5).astype(np.uint8)
    elif CFG.mask_fusion_method == "intersection":
        return np.logical_and(rgb_mask > 0.5, thermal_mask > 0.5).astype(np.uint8)
    else:
        combined = CFG.mask_rgb_weight * rgb_mask + CFG.mask_thermal_weight * thermal_mask
        return (combined > 0.5).astype(np.uint8)

# ── IR simulation ──────────────────────────────────────────────────────
def rgb_to_ir(rgb: np.ndarray) -> np.ndarray:
    """
    Simulate infrared camera output from an RGB frame.
    Matches the look in your reference image:
      - Dark sky / background
      - Bright heat sources (vehicles, headlights)
      - High local contrast (CLAHE)
    Returns a 3-channel BGR image that looks grayscale (like real IR).
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    # Invert: bright → hot, dark → cold  (standard IR look)
    inverted = cv2.bitwise_not(gray)

    # CLAHE — local contrast enhancement, makes vehicles pop
    clahe    = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(inverted)

    # Convert back to 3-channel BGR (YOLO needs 3 channels)
    ir_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return ir_bgr

# ── Core fusion ────────────────────────────────────────────────────────
def run_fusion(rgb: np.ndarray, ir: np.ndarray) -> dict:
    rgb_model, ir_model = get_models()
    coco_to_id = {v: k for k, v in rgb_model.names.items()}

    t0 = time.perf_counter()

    # 1. Inference
    rgb_res = rgb_model(rgb, verbose=False, conf=CFG.conf)[0]
    ir_res  = ir_model( ir,  verbose=False, conf=CFG.conf)[0]

    # Pre-extract original masks & boxes
    rgb_masks_all = rgb_res.masks.data.cpu().numpy() if rgb_res.masks is not None else []
    rgb_boxes_all = [b.xyxy[0].cpu().numpy() for b in rgb_res.boxes]
    th_masks_all  = ir_res.masks.data.cpu().numpy()  if ir_res.masks is not None else []
    th_boxes_all  = []
    th_classes_all = []

    # Filter IR boxes to COCO subset to match RGB indices later
    pool_th_boxes, pool_th_scores, pool_th_classes = [], [], []
    for i, b in enumerate(ir_res.boxes):
        ir_label = ir_model.names[int(b.cls)]
        # Robust lookup: try mapping first, then check if it's already a COCO model
        coco_label = IR_TO_COCO.get(ir_label)
        if coco_label:
            cls_id = coco_to_id.get(coco_label, -1)
        else:
            cls_id = coco_to_id.get(ir_label, -1)

        th_boxes_all.append(b.xyxy[0].cpu().numpy())
        th_classes_all.append(cls_id)
        if cls_id != -1:
            pool_th_boxes.append(b.xyxy[0].tolist())
            pool_th_scores.append(float(b.conf))
            pool_th_classes.append(cls_id)

    rgb_b = np.array([b.xyxy[0].cpu().numpy() for b in rgb_res.boxes]) if len(rgb_res.boxes) > 0 else np.empty((0,4))
    rgb_s = np.array([float(b.conf)           for b in rgb_res.boxes]) if len(rgb_res.boxes) > 0 else np.array([])
    rgb_c = np.array([int(b.cls)              for b in rgb_res.boxes]) if len(rgb_res.boxes) > 0 else np.array([], dtype=int)

    th_b  = np.array(pool_th_boxes) if pool_th_boxes else np.empty((0,4))
    th_s  = np.array(pool_th_scores) if pool_th_scores else np.array([])
    th_c  = np.array(pool_th_classes, dtype=int) if pool_th_classes else np.array([], dtype=int)

    # 2. Apply Custom Per-Class Confidence Thresholds
    rgb_b, rgb_s, rgb_c = apply_class_conf_filter(rgb_b, rgb_s, rgb_c, rgb_model.names)
    th_b, th_s, th_c    = apply_class_conf_filter(th_b, th_s, th_c, rgb_model.names)

    # 3. IoU Union Fusion
    fused_b, fused_s, fused_c, fused_src = fuse_detections_union(
        rgb_b, rgb_s, rgb_c, th_b, th_s, th_c, iou_thresh=NMS_IOU
    )

    # 4. Tracking
    fusion_count = len(fused_b)
    track_ids = [None] * fusion_count
    if fusion_count > 0:
        detections = sv.Detections(
            xyxy=fused_b,
            confidence=fused_s,
            class_id=fused_c,
        )
        tracked_det = tracker.update_with_detections(detections)
        # Map tracked IDs back to original fused boxes using IoU
        if len(tracked_det) > 0:
            for i, tb in enumerate(tracked_det.xyxy):
                ious = [iou(tb, fb) for fb in fused_b]
                best_idx = int(np.argmax(ious))
                if ious[best_idx] > 0.5:
                    track_ids[best_idx] = int(tracked_det.tracker_id[i])

    # Helper to extract polygon from mask array
    def get_poly(mask_arr, w, h):
        if mask_arr is None: return None
        # Convert to uint8 for OpenCV compatibility
        m_uint8 = (mask_arr > 0.5).astype(np.uint8)
        m = cv2.resize(m_uint8, (w, h))
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        largest = max(contours, key=cv2.contourArea)
        epsilon = 0.005 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        return approx.reshape(-1, 2).tolist()

    def find_best_iou(query_box, candidate_boxes, iou_thresh=0.30):
        best_val, best_idx = 0.0, -1
        for k, cb in enumerate(candidate_boxes):
            v = iou(query_box, cb)
            if v > best_val:
                best_val, best_idx = v, k
        return best_idx if best_val >= iou_thresh else -1

    w, h = rgb.shape[1], rgb.shape[0]

    # 5. Build Fusion Output List
    fusion_list: list[Box] = []
    fusion_masks = []

    for i in range(fusion_count):
        x1, y1, x2, y2 = fused_b[i]
        cls      = int(fused_c[i])
        conf     = float(fused_s[i])
        track_id = track_ids[i]
        label    = rgb_model.names[cls]
        det_box  = [x1, y1, x2, y2]

        r_idx = find_best_iou(det_box, rgb_boxes_all)
        t_idx = find_best_iou(det_box, th_boxes_all)

        src_str = "fusion"
        if r_idx >= 0 and t_idx == -1: src_str = "rgb"
        if t_idx >= 0 and r_idx == -1: src_str = "ir"

        # Fuse masks
        final_mask_poly = None
        if r_idx >= 0 and t_idx >= 0 and len(rgb_masks_all) > r_idx and len(th_masks_all) > t_idx:
            mask_r = cv2.resize(rgb_masks_all[r_idx], (w, h))
            mask_t = cv2.resize(th_masks_all[t_idx],  (w, h))
            final_mask = fuse_masks(mask_r, mask_t)
            final_mask_poly = get_poly(final_mask, w, h)
            fusion_masks.append(final_mask)
        elif r_idx >= 0 and len(rgb_masks_all) > r_idx:
            mask_r = cv2.resize(rgb_masks_all[r_idx], (w, h))
            final_mask_poly = get_poly(mask_r > 0.5, w, h)
            fusion_masks.append(mask_r > 0.5)
        elif t_idx >= 0 and len(th_masks_all) > t_idx:
            mask_t = cv2.resize(th_masks_all[t_idx], (w, h))
            final_mask_poly = get_poly(mask_t > 0.5, w, h)
            fusion_masks.append(mask_t > 0.5)

        # Distance estimation
        raw_dist, _ = estimate_distance(label, y2 - y1)
        dist = None
        if raw_dist:
            dist = smooth_distance(track_id, raw_dist) if track_id is not None else raw_dist

        fusion_list.append(Box(
            label=label, conf=round(conf, 3),
            x1=x1, y1=y1, x2=x2, y2=y2,
            source=src_str, polygon=final_mask_poly,
            track_id=track_id, distance=round(dist, 2) if dist else None
        ))

    # Single modality lists
    rgb_list = []
    for i, b in enumerate(rgb_b):
        p = get_poly(rgb_masks_all[i], w, h) if i < len(rgb_masks_all) else None
        rgb_list.append(Box(label=rgb_model.names[int(rgb_c[i])], conf=round(float(rgb_s[i]), 3),
            x1=b[0], y1=b[1], x2=b[2], y2=b[3], source="rgb", polygon=p))
            
    ir_list = []
    for i, b in enumerate(th_b):
        p = get_poly(th_masks_all[i], w, h) if i < len(th_masks_all) else None
        ir_list.append(Box(label=rgb_model.names[int(th_c[i])], conf=round(float(th_s[i]), 3),
            x1=b[0], y1=b[1], x2=b[2], y2=b[3], source="ir", polygon=p))

    ms  = round((time.perf_counter() - t0) * 1000, 1)
    avg = lambda lst: round(float(np.mean([d.conf for d in lst])), 3) if lst else 0.0

    return dict(
        rgb_detections=rgb_list,
        ir_detections=ir_list,
        fusion_detections=fusion_list,
        rgb_count=len(rgb_list),
        ir_count=len(ir_list),
        fusion_count=len(fusion_list),
        avg_conf_rgb=avg(rgb_list),
        avg_conf_ir=avg(ir_list),
        avg_conf_fusion=avg(fusion_list),
        processing_ms=ms,
        device=DEVICE,
        _rgb_masks=rgb_masks_all if len(rgb_masks_all) > 0 else None,
        _ir_masks=th_masks_all if len(th_masks_all) > 0 else None,
        _fusion_masks=fusion_masks if fusion_masks else None,
    )

# ── Drawing ────────────────────────────────────────────────────────────
def draw_boxes(img: np.ndarray, boxes: list[Box], color: tuple, title: str, masks=None) -> np.ndarray:
    out = img.copy()
    # Draw segmentation masks first (under boxes)
    if masks is not None:
        overlay = out.copy()
        for mask in masks:
            # Ensure mask is uint8 before resize
            m_uint8 = (mask > 0.5).astype(np.uint8)
            mask_resized = cv2.resize(m_uint8, (img.shape[1], img.shape[0]))
            overlay[mask_resized > 0] = (
                0.5 * overlay[mask_resized > 0] + 0.5 * np.array(color)
            ).astype(np.uint8)
        out = overlay
    for b in boxes:
        x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)
        label = f"{b.label} {b.conf:.2f}"
        if b.distance is not None:
            label += f" [{b.distance:.2f}m]"
        if b.track_id is not None:
            label = f"ID:{b.track_id} {label}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(out, title, (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)
    return out

def decode(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Cannot decode image")
    return img

def to_jpeg(img: np.ndarray, q: int = 88) -> bytes:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes()

# ── Routes ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "rgb_model_loaded": _rgb_model is not None,
        "ir_model_loaded":  _ir_model  is not None,
        "rgb_model_path": RGB_MODEL_PATH,
        "ir_model_path":  IR_MODEL_PATH,
    }


@app.post("/detect/image", response_model=FusionResult)
async def detect_image(
    rgb_file: UploadFile = File(..., description="RGB camera image"),
    ir_file:  Optional[UploadFile] = File(None, description="IR image (auto-generated if omitted)"),
):
    rgb = decode(await rgb_file.read())
    ir  = decode(await ir_file.read()) if ir_file else rgb_to_ir(rgb)
    if ir_file:
        ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]))
    return FusionResult(**run_fusion(rgb, ir))


@app.post("/detect/image/visualize")
async def visualize_image(
    rgb_file: UploadFile = File(...),
    ir_file:  Optional[UploadFile] = File(None),
):
    """Returns a side-by-side annotated JPEG: [ RGB | IR | FUSION ]"""
    rgb = decode(await rgb_file.read())
    ir  = decode(await ir_file.read()) if ir_file else rgb_to_ir(rgb)
    if ir_file:
        ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]))

    result = run_fusion(rgb, ir)
    combined = np.hstack([
        draw_boxes(rgb, result["rgb_detections"],     (255, 80,  0), "RGB",
                   masks=result.get("_rgb_masks")),
        draw_boxes(ir,  result["ir_detections"],      (0,  220, 60), "IR",
                   masks=result.get("_ir_masks")),
        draw_boxes(ir,  result["fusion_detections"],  (0,  30, 220), "FUSION",
                   masks=result.get("_fusion_masks")),
    ])
    return StreamingResponse(io.BytesIO(to_jpeg(combined)), media_type="image/jpeg")


@app.post("/detect/frame", response_model=FusionResult)
async def detect_frame(
    frame:    UploadFile = File(...),
    ir_frame: Optional[UploadFile] = File(None),
):
    """Lightweight per-frame endpoint for live video streaming."""
    return await detect_image(frame, ir_frame)


@app.post("/video/process")
async def process_video(
    bg: BackgroundTasks,
    video_file: UploadFile = File(...),
    ir_video_file: Optional[UploadFile] = File(None, description="IR video (auto-generated if omitted)"),
):
    """Queue a full-video batch job. Poll /video/status/{job_id}."""
    job_id   = str(uuid.uuid4())
    ext      = Path(video_file.filename or "v.mp4").suffix or ".mp4"
    in_path  = Path("uploads") / f"{job_id}_in{ext}"
    out_path = Path("outputs") / f"{job_id}_fusion.mp4"
    in_path.write_bytes(await video_file.read())

    ir_in_path = None
    if ir_video_file:
        ir_ext     = Path(ir_video_file.filename or "ir.mp4").suffix or ".mp4"
        ir_in_path = Path("uploads") / f"{job_id}_ir_in{ir_ext}"
        ir_in_path.write_bytes(await ir_video_file.read())

    video_jobs[job_id] = dict(status="queued", progress=0.0,
                               frames_done=0, total_frames=0,
                               output_path=str(out_path), error=None)
    bg.add_task(_run_video_job, job_id, str(in_path), str(out_path),
                str(ir_in_path) if ir_in_path else None)
    return {"job_id": job_id, "status": "queued"}


@app.get("/video/status/{job_id}", response_model=JobStatus)
def video_status(job_id: str):
    if job_id not in video_jobs:
        raise HTTPException(404, "Job not found")
    return JobStatus(job_id=job_id, **video_jobs[job_id])


@app.get("/video/download/{job_id}")
def video_download(job_id: str):
    if job_id not in video_jobs:
        raise HTTPException(404, "Job not found")
    j = video_jobs[job_id]
    if j["status"] != "done":
        raise HTTPException(400, f"Not ready (status={j['status']})")
    return FileResponse(j["output_path"], media_type="video/mp4",
                        filename=f"fusion_{job_id[:8]}.mp4")


# ── Background video processor ─────────────────────────────────────────
def _run_video_job(job_id: str, in_path: str, out_path: str, ir_in_path: Optional[str] = None):
    job = video_jobs[job_id]
    job["status"] = "processing"
    ir_cap = None
    try:
        cap   = cv2.VideoCapture(in_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 20
        W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        job["total_frames"] = total

        if ir_in_path:
            ir_cap = cv2.VideoCapture(ir_in_path)

        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W * 3, H))
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok: break

            if ir_cap:
                ir_ok, ir_frame = ir_cap.read()
                if ir_ok:
                    ir = cv2.resize(ir_frame, (W, H))
                else:
                    ir = rgb_to_ir(frame)  # fallback if IR video is shorter
            else:
                ir = rgb_to_ir(frame)

            res    = run_fusion(frame, ir)
            writer.write(np.hstack([
                draw_boxes(frame, res["rgb_detections"],     (255, 80,  0), "RGB",
                           masks=res.get("_rgb_masks")),
                draw_boxes(ir,    res["ir_detections"],      (0,  220, 60), "IR",
                           masks=res.get("_ir_masks")),
                draw_boxes(ir,    res["fusion_detections"],  (0,  30, 220), "FUSION",
                           masks=res.get("_fusion_masks")),
            ]))
            idx += 1
            job["frames_done"] = idx
            job["progress"]    = round(idx / max(total, 1) * 100, 1)

        cap.release()
        if ir_cap: ir_cap.release()
        writer.release()
        job["status"] = "done"; job["progress"] = 100.0
    except Exception as e:
        job["status"] = "error"; job["error"] = str(e); raise
    finally:
        try: os.remove(in_path)
        except OSError: pass
        if ir_in_path:
            try: os.remove(ir_in_path)
            except OSError: pass