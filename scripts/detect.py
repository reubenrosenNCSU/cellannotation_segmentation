# detect.py

import os
import json
import argparse
import sys
from PIL import Image
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import LOGGER
LOGGER.setLevel("ERROR")


def normalize_image_percentile(img_path, percentile_low=1, percentile_high=99):
    """Normalize 16-bit single-channel TIFF image using percentile clipping."""
    img = Image.open(img_path)
    arr = np.array(img)

    if arr.dtype != np.uint16:
        raise ValueError("Expected 16-bit image")

    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    elif arr.ndim != 2:
        raise ValueError("Expected single-channel image")

    p_low = np.percentile(arr, percentile_low)
    p_high = np.percentile(arr, percentile_high)
    arr_clipped = np.clip(arr, p_low, p_high)
    arr_normalized = ((arr_clipped - p_low) * 255.0 / (p_high - p_low)).astype(np.uint8)

    return Image.fromarray(arr_normalized)


def convert_image_for_detection(img_path):
    """Use percentile-based histogram normalization and return RGB image."""
    img = normalize_image_percentile(img_path)
    return img.convert('RGB')  # Ensure it's RGB

def detect(image_path, model_path, threshold):
    img = Image.open(image_path)
    width, height = img.size
    img = convert_image_for_detection(img)

    model = YOLO(model_path)
    results = model.predict(source=img, conf=threshold, save=False, save_txt=False, save_conf=True)

    annotations = []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls.item())
            x_center, y_center, w, h = box.xywhn[0].cpu().numpy()
            annotations.append(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    return {
        "image_width": width,
        "image_height": height,
        "annotations": "\n".join(annotations)
    }


if __name__ == "__main__":
    import argparse
    import json
    import sys
    from ultralytics.utils import LOGGER  # <--- ADD THIS

    LOGGER.setLevel("ERROR")  # <--- Suppress all Ultralytics logs

    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to the image file')
    parser.add_argument('--model', required=True, help='Path to the YOLO model (.pt)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold')
    args = parser.parse_args()

    try:
        result = detect(args.image, args.model, args.threshold)
        print(json.dumps(result))  # JSON-only output
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
