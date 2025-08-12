import os
import time
import yaml
import shutil
from glob import glob
from ultralytics import YOLO
from sklearn.model_selection import KFold
from PIL import Image

def convert_pre_annotations(pre_csv_path, dest_label_dir, classes_map):
    import csv
    annos = {}
    with open(pre_csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 6: continue
            fname, x1, y1, x2, y2, cls = row
            annos.setdefault(fname, []).append((float(x1),float(y1),float(x2),float(y2),classes_map[int(cls)]))
    for fname, boxes in annos.items():
        txtname = os.path.splitext(fname)[0] + '.txt'
        outpath = os.path.join(dest_label_dir, txtname)
        with open(outpath, 'w') as out:
            for x1,y1,x2,y2,cls in boxes:
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                out.write(f"{cls} {cx} {cy} {w} {h}\n")

def convert_tiff_to_png(tiff_path, output_dir):
    base = os.path.splitext(os.path.basename(tiff_path))[0]
    png_path = os.path.join(output_dir, base + '.png')
    with Image.open(tiff_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(png_path)
    return png_path

def run_kfold(base_path, user_id, model_type, model_map, folds=5, epochs=10, img_ext='*.png', classes=None):
    saved_data = os.path.join(base_path, user_id, 'images')
    saved_labels = os.path.join(base_path, user_id, 'labels')

    # Convert pre_annot.csv to YOLO format
    pre_folder = os.path.join(base_path, f'pre_{model_type.lower()}')
    pre_csv = os.path.join(pre_folder, 'pre_annot.csv')
    if os.path.exists(pre_csv):
        convert_pre_annotations(pre_csv, saved_labels, {i: i for i in range(len(classes))})
        for img_path in glob(os.path.join(pre_folder, '*.tif')):
            png_path = convert_tiff_to_png(img_path, saved_data)

    model_name = model_map.get(model_type, 'yolov11n.pt')
    img_list = sorted(os.path.basename(p) for p in glob(os.path.join(saved_data, img_ext)))
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)

    best_mAP = 0.0
    best_pt = None
    mAPs = []

    for fold_idx, (train_ix, val_ix) in enumerate(kf.split(img_list), start=1):
        data_yaml = {
            'path': os.path.join(base_path, user_id),
            'train': 'images',
            'val':   'images',
            'nc':    len(classes),
            'names': classes
        }
        model = YOLO(model_name)
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=640,
            batch=8,
            project=f'runs/{user_id}/cv',
            name=f'fold_{fold_idx}',
            exist_ok=True
        )
        mAP50 = getattr(results.metrics, 'map50', 0.0)
        mAPs.append(mAP50)
        pt_path = os.path.join(results.save_dir, 'weights', 'best.pt')
        if mAP50 > best_mAP:
            best_mAP = mAP50
            best_pt = pt_path

    avg_mAP = sum(mAPs) / len(mAPs)
    time.sleep(2)
    return avg_mAP, best_pt
