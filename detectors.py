# detectors.py

from qrdet import QRDetector
from ultralytics import YOLO
from pdf2image import convert_from_bytes
from PIL import Image, ImageDraw
import numpy as np
import cv2
import json
import os

# ============================================
# 1. PORTABLE PATH DETECTION
# ============================================

# Folder where THIS file lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder for model weights: <project>/models
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Expected weight files inside /models
SEAL_WEIGHTS_PATH = os.path.join(MODELS_DIR, "seals.pt")
SIGN_WEIGHTS_PATH = os.path.join(MODELS_DIR, "signatures.pt")

print("Seal weights:", SEAL_WEIGHTS_PATH, "exists =", os.path.exists(SEAL_WEIGHTS_PATH))
print("Signature weights:", SIGN_WEIGHTS_PATH, "exists =", os.path.exists(SIGN_WEIGHTS_PATH))

# ============================================
# 2. MODEL INITIALIZATION
# ============================================

# QRDet: YOLO-based QR code detector
qr_detector = QRDetector(
    model_size="s",
    conf_th=0.5,
    nms_iou=0.3
)

# YOLO models for seals and signatures
seal_model = YOLO(SEAL_WEIGHTS_PATH)
signature_model = YOLO(SIGN_WEIGHTS_PATH)

# ============================================
# 3. HELPER FUNCTIONS
# ============================================

def pdf_to_images(pdf_bytes, dpi=200):
    """Convert PDF byte data into a list of PIL.PngImagePlugin.Image objects."""
    return convert_from_bytes(pdf_bytes, dpi=dpi)


def detect_qr_codes(pil_image):
    """Detect QR codes using QRDet."""
    img_rgb = np.array(pil_image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    detections = qr_detector.detect(image=img_bgr, is_bgr=True)

    boxes = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        conf = float(det["confidence"])
        boxes.append({
            "type": "qr_code",
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "confidence": conf,
        })
    return boxes


def detect_with_yolo_single_model(pil_image, model, object_type, conf=0.25):
    """Detect 'seal' or 'signature' using the YOLO model passed in."""
    img_np = np.array(pil_image)
    results = model.predict(source=img_np, conf=conf, verbose=False)
    res = results[0]

    boxes = []
    if res.boxes:
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i].tolist()
            boxes.append({
                "type": object_type,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "confidence": float(confs[i]),
                "class_id": int(clss[i]),
            })
    return boxes


def draw_all_boxes(base_image, detections):
    """Draw bounding boxes on the page image."""
    color_map = {
        "qr_code": (0, 255, 0),      # green
        "seal": (255, 0, 0),         # red
        "signature": (0, 0, 255),    # blue
    }

    img = base_image.copy()
    draw = ImageDraw.Draw(img)

    for det in detections:
        x1 = int(det["x1"])
        y1 = int(det["y1"])
        x2 = int(det["x2"])
        y2 = int(det["y2"])
        label = det["type"]
        conf = det["confidence"]

        color = color_map.get(label, (255, 255, 0))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        text = f"{label} {conf:.2f}"
        draw.text((x1 + 2, max(y1 - 12, 0)), text, fill=color)

    return img

# ============================================
# 4. MAIN DETECTION PIPELINE
# ============================================

def detect_in_pdf(pdf_file, conf_yolo=0.25, dpi=200):
    """
    Process PDF through QRDet + YOLO (seals + signatures).
    Returns:
        results_per_page: page images + boxes
        json_str: flat JSON of all detections
    """
    pdf_bytes = pdf_file.read()
    pages = pdf_to_images(pdf_bytes, dpi=dpi)

    results_per_page = []
    all_boxes_flat = []

    for idx, page in enumerate(pages):
        page_index = idx + 1

        qr_boxes = detect_qr_codes(page)
        seal_boxes = detect_with_yolo_single_model(page, seal_model, "seal", conf_yolo)
        sig_boxes = detect_with_yolo_single_model(page, signature_model, "signature", conf_yolo)

        all_boxes = qr_boxes + seal_boxes + sig_boxes
       

        for b in all_boxes:
            all_boxes_flat.append({"page": page_index, **b})

        annotated = draw_all_boxes(page, all_boxes)

        results_per_page.append({
            "page_index": page_index,
            "all_boxes": all_boxes,
            "image_with_boxes": annotated,
        })

    json_str = json.dumps(all_boxes_flat, indent=2)
    return results_per_page, json_str
