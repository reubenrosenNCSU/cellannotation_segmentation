# scripts/kfold_train.py
import os, shutil, argparse
from sklearn.model_selection import KFold
from ultralytics import YOLO

def get_dataset(image_dir, label_dir):
    samples = []
    for fname in os.listdir(image_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            base = os.path.splitext(fname)[0]
            label_file = os.path.join(label_dir, f"{base}.txt")
            if os.path.exists(label_file):
                samples.append((os.path.join(image_dir, fname), label_file))
    return samples

def write_yaml(path, nc, names, root_dir):
    with open(path, 'w') as f:
        f.write(f"path: {root_dir}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write(f"nc: {nc}\n")
        f.write(f"names: {names}\n")

def main(args):
    samples = get_dataset(args.image_dir, args.label_dir)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for i, (train_idx, val_idx) in enumerate(kf.split(samples)):
        fold_dir = os.path.join(args.output_dir, f"fold_{i}")
        train_img = os.path.join(fold_dir, 'train', 'images')
        train_lbl = os.path.join(fold_dir, 'train', 'labels')
        val_img = os.path.join(fold_dir, 'val', 'images')
        val_lbl = os.path.join(fold_dir, 'val', 'labels')
        for p in [train_img, train_lbl, val_img, val_lbl]:
            os.makedirs(p, exist_ok=True)

        for idx in train_idx:
            shutil.copy2(samples[idx][0], train_img)
            shutil.copy2(samples[idx][1], train_lbl)
        for idx in val_idx:
            shutil.copy2(samples[idx][0], val_img)
            shutil.copy2(samples[idx][1], val_lbl)

        yaml_path = os.path.join(fold_dir, 'data.yaml')
        write_yaml(yaml_path, args.nc, args.names, fold_dir)

        model = YOLO(args.weights)
        model.train(data=yaml_path, epochs=args.epochs, imgsz=640, batch=4, name=f"fold_{i}", project=fold_dir)
        val_result = model.val(data=yaml_path)
        map50 = val_result.box.map50
        results.append(map50)
        print(f"[FOLD {i}] mAP@0.5 = {map50:.4f}")

    avg_map = sum(results) / len(results)
    print(f"[K-FOLD] Average mAP@0.5 = {avg_map:.4f}")

    with open(os.path.join(args.output_dir, 'kfold_results.txt'), 'w') as f:
        for i, val in enumerate(results):
            f.write(f"Fold {i}: mAP@0.5 = {val:.4f}\n")
        f.write(f"\nAverage mAP@0.5 = {avg_map:.4f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--label_dir', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--nc', type=int, required=True)
    parser.add_argument('--names', nargs='+', required=True)
    args = parser.parse_args()
    main(args)
