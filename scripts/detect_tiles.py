# detect_tiles.py
import os
from PIL import Image
from ultralytics import YOLO
import numpy as np

def convert_image_for_detection(img):
    if img.mode != 'RGB':
        if img.mode.startswith('I;16'):
            arr = np.array(img).astype(np.uint16)
            min_val = arr.min()
            max_val = arr.max()
            if max_val > min_val:
                arr = ((arr - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                arr = (arr // 256).astype(np.uint8)
            img = Image.fromarray(arr)
        img = img.convert('RGB')
    return img

def detect_tiles_in_batch(tiles_dir, output_dir, model_path, threshold):
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_path)

    for fname in os.listdir(tiles_dir):
        if not fname.endswith('.png'):
            continue

        tile_path = os.path.join(tiles_dir, fname)
        output_path = os.path.join(output_dir, fname.replace('.png', '.txt'))

        try:
            img = Image.open(tile_path)
            width, height = img.size
            img = convert_image_for_detection(img)

            results = model.predict(source=img, conf=threshold, save=False)

            with open(output_path, 'w') as f:
                for result in results:
                    for box in result.boxes:
                        cls = int(box.cls.item())
                        x_center, y_center, w, h = box.xywhn[0].cpu().numpy()
                        f.write(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

        except Exception as e:
            print(f"Error on tile {fname}: {str(e)}")
