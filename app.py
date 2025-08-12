from werkzeug.utils import secure_filename  # ADD THIS AT TOP OF FILE
from flask import Flask, request, jsonify, send_from_directory, send_file
import os
from PIL import Image
import uuid
from flask_cors import CORS
import subprocess
import shutil
import h5py
from PIL import Image, ImageOps
import numpy as np
import zipfile
import io
import gc  # Garbage collector
import time  # For delays
from flask import session
from datetime import timedelta
from apscheduler.schedulers.background import BackgroundScheduler
import datetime
import atexit
from tensorflow.python.summary.summary_iterator import summary_iterator
import glob
import tensorflow as tf
from ultralytics import YOLO
from scripts.normalization import normalize_image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Gets directory where app.py is

def sanitize_box(x1, y1, x2, y2):
    """Ensure x1 <= x2 and y1 <= y2"""
    new_x1 = min(x1, x2)
    new_y1 = min(y1, y2)
    new_x2 = max(x1, x2)
    new_y2 = max(y1, y2)
    return new_x1, new_y1, new_x2, new_y2

def get_scalar_value(v):
    if hasattr(v, 'simple_value') and v.simple_value != 0.0:
        return v.simple_value
    elif hasattr(v, 'tensor'):
        try:
            t = tf.make_ndarray(v.tensor)
            return float(t)
        except Exception as e:
            print(f"[Error extracting tensor value for {v.tag}]: {e}")
            return None
    return None

app = Flask(__name__)
CORS(app, supports_credentials=True)  # This will allow all domains to access your API

app.secret_key = 'test'  # Replace with a real secret key


@app.before_request
def set_user_session():
    if 'user_id' not in session:
        session.permanent = True
        session['user_id'] = str(uuid.uuid4())
    
    # Always update directory modification time on any request
    user_id = session['user_id']
    user_dir = os.path.join('users', user_id)
    if os.path.exists(user_dir):
        os.utime(user_dir, None)  # Update mtime on every request
    # Create user directories if they don't exist
    user_id = session['user_id']
    user_dirs = [
        os.path.join('users', user_id, 'uploads'),
        os.path.join('users', user_id, 'converted'),
        os.path.join('users', user_id, 'saved_data'),
        os.path.join('users', user_id, 'saved_annotations'),
        os.path.join('users', user_id, 'finaloutput'),
        os.path.join('users', user_id, 'ft_upload'),
        os.path.join('users', user_id, 'images'),
        os.path.join('users', user_id, 'input'),
        os.path.join('users', user_id, 'output'),
        os.path.join('users', user_id, 'output/output_csv'),
        os.path.join('users', user_id, 'snapshots')
    ]
    for dir_path in user_dirs:
        os.makedirs(dir_path, exist_ok=True)

app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)  # Session expires after 24 hours

@app.route('/cleanup', methods=['POST'])
def cleanup_files():
    try:
        user_id = session.get('user_id')
        if user_id:
            user_dir = os.path.join('users', user_id)
            if os.path.exists(user_dir):
                shutil.rmtree(user_dir)
                print(f"Cleaned up directory for user: {user_id}")
        return jsonify({'status': 'success'})
    except Exception as e:
        print(f"Cleanup error: {str(e)}")
        return jsonify({'status': 'error'}), 500



# Add these additional directories to clear
CLEANUP_DIRS = ['output', 'input', 'images']

# Create directories if they don't exist
def clear_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            clear_folder(file_path)



def clear_uploaded_images():
    """Delete all files in the current user's upload folder"""
    user_id = session.get('user_id', 'default')  # Handle unauthenticated edge case
    user_upload_dir = os.path.join('users', user_id, 'uploads')
    for filename in os.listdir(user_upload_dir):
        file_path = os.path.join(user_upload_dir, filename)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")



@app.route('/upload', methods=['POST'])
def upload_file():
    user_id = session['user_id']
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        clear_uploaded_images()
        original_name = file.filename
        base_name = os.path.splitext(original_name)[0]
        original_extension = os.path.splitext(original_name)[1][1:].lower()
        user_upload_dir = os.path.join('users', user_id, 'uploads')
        user_converted_dir = os.path.join('users', user_id, 'converted')

        # Save original file
        original_path = os.path.join(user_upload_dir, original_name)
        file.save(original_path)

        # Store original dimensions
        with Image.open(original_path) as img:
            session['original_dimensions'] = img.size
            session['current_dimensions'] = img.size
            session['target_diameter'] = 34.0

        # Generate normalized preview
        unique_id = str(uuid.uuid4())
        output_filename = f"{unique_id}.png"
        output_path = os.path.join(user_converted_dir, output_filename)
        
        normalize_image(original_path, output_path)

        return jsonify({
            'converted_url': f'/converted/{output_filename}',
            'original_name': original_name,
            'base_name': base_name,
            'original_extension': original_extension
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    
@app.route('/export-annotations', methods=['POST'])
def export_annotations():
    user_id = session['user_id']
    try:
        # Get YOLO segmentation data
        data = request.json
        yolo_data = data['yolo_data']
        original_filename = data['original_filename']
        user_upload_dir = os.path.join('users', user_id, 'uploads')

        # Get current TIFF path
        tiff_path = os.path.join(user_upload_dir, original_filename)
        if not os.path.exists(tiff_path):
            return jsonify({'error': 'Current TIFF file not found'}), 404

        # Create in-memory ZIP file
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add YOLO txt file
            txt_filename = f"{os.path.splitext(original_filename)[0]}.txt"
            zipf.writestr(txt_filename, yolo_data)
            # Add TIFF
            zipf.write(tiff_path, os.path.basename(tiff_path))

        zip_buffer.seek(0)

        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'{os.path.splitext(original_filename)[0]}_export.zip'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload-cropped', methods=['POST'])
def upload_cropped_file():
    user_id = session['user_id']
    try:
        # Get crop coordinates and original filename
        original_name = request.form['original_filename']
        x = int(float(request.form['x']))
        y = int(float(request.form['y']))
        width = int(float(request.form['width']))
        height = int(float(request.form['height']))
        user_upload_dir = os.path.join('users', user_id, 'uploads')
        user_convert_dir = os.path.join('users', user_id, 'converted')

        # Path to original TIFF
        upload_path = os.path.join(user_upload_dir, original_name)
        
        # Open and crop original image
        with Image.open(upload_path) as img:
            # Perform crop on original TIFF
            cropped_img = img.crop((x, y, x + width, y + height))
            
            # Overwrite original file with cropped version
            cropped_img.save(upload_path, format='TIFF', compression='tiff_deflate')
            # 🔄 Update original and current dimensions to CROPPED size
            session['original_dimensions'] = cropped_img.size  # (new_width, new_height)
            session['current_dimensions'] = cropped_img.size


        # Generate new PNG preview from updated TIFF
        unique_id = str(uuid.uuid4())
        output_filename = f"{unique_id}.png"
        output_path = os.path.join(user_convert_dir, output_filename)
        cropped_img.save(output_path, "PNG")

        return jsonify({
            'converted_url': f'/converted/{output_filename}',
            'original_name': original_name,  # Keep original filename
            'base_name': os.path.splitext(original_name)[0],
            'original_extension': 'tiff'
        })

    except Exception as e:
        print(f"Error in upload-cropped: {str(e)}")
        return jsonify({'error': f"Server error: {str(e)}"}), 500
    
from scripts.detect_tiles import detect_tiles_in_batch

@app.route('/detect-sgn', methods=['POST'])
def detect_sgn():
    import json

    user_id = session['user_id']
    threshold = float(request.json.get('threshold', 0.5))
    upload_dir = os.path.join('users', user_id, 'uploads')
    tiles_dir = os.path.join('users', user_id, 'images', 'tiles')
    output_txt_dir = os.path.join('users', user_id, 'images', 'tiles_output')
    merged_output_path = os.path.join('users', user_id, 'finaloutput', 'merged_sgn.txt')

    os.makedirs(tiles_dir, exist_ok=True)
    os.makedirs(output_txt_dir, exist_ok=True)

    try:
        files = [f for f in os.listdir(upload_dir) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))]
        if not files:
            return jsonify({'error': 'No image found. Upload an image first.'}), 400

        image_path = os.path.join(upload_dir, files[0])
        model_path = 'snapshots/SGN_best.pt'

        # Step 1: Split into tiles
        subprocess.run(['python3', 'scripts/split_image.py', '--image', image_path, '--output', tiles_dir], check=True)

        # Step 2: Detect all tiles using GPU-efficient function
        detect_tiles_in_batch(tiles_dir, output_txt_dir, model_path, threshold)

        # Get image size from original
        with Image.open(image_path) as img:
            image_width, image_height = img.size

        # Step 3: Merge annotations
        subprocess.run([
            'python3', 'scripts/merge_annotations.py',
            '--tiles', output_txt_dir,
            '--output', merged_output_path,
            '--image_width', str(image_width),
            '--image_height', str(image_height)
        ], check=True)

        with open(merged_output_path, 'r') as f:
            final_annotation = f.read()

        return jsonify({
            "annotations": final_annotation,
            "image_width": image_width,
            "image_height": image_height
        })

    except Exception as e:
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500

@app.route('/detect-cd3', methods=['POST'])
def detect_cd3():
    user_id = session['user_id']
    threshold = float(request.json.get('threshold', 0.5))
    upload_dir = os.path.join('users', user_id, 'uploads')
    tiles_dir = os.path.join('users', user_id, 'images', 'tiles')
    output_txt_dir = os.path.join('users', user_id, 'images', 'tiles_output')
    merged_output_path = os.path.join('users', user_id, 'finaloutput', 'merged_cd3.txt')

    os.makedirs(tiles_dir, exist_ok=True)
    os.makedirs(output_txt_dir, exist_ok=True)

    try:
        files = [f for f in os.listdir(upload_dir) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))]
        if not files:
            return jsonify({'error': 'No image found. Upload an image first.'}), 400

        image_path = os.path.join(upload_dir, files[0])
        model_path = 'snapshots/cd3_v2.pt'  # Changed model path

        # Split into tiles
        subprocess.run(['python3', 'scripts/split_image.py', '--image', image_path, '--output', tiles_dir], check=True)

        # Detect all tiles
        detect_tiles_in_batch(tiles_dir, output_txt_dir, model_path, threshold)

        # Get image size
        with Image.open(image_path) as img:
            image_width, image_height = img.size

        # Merge annotations
        subprocess.run([
            'python3', 'scripts/merge_annotations.py',
            '--tiles', output_txt_dir,
            '--output', merged_output_path,
            '--image_width', str(image_width),
            '--image_height', str(image_height)
        ], check=True)

        with open(merged_output_path, 'r') as f:
            final_annotation = f.read()

        return jsonify({
            "annotations": final_annotation,
            "image_width": image_width,
            "image_height": image_height
        })

    except Exception as e:
        return jsonify({'error': f'CD3 detection failed: {str(e)}'}), 500




@app.route('/converted/<filename>')
def serve_converted(filename):
    user_id = session['user_id']
    converted_dir = os.path.join('users', user_id, 'converted')
    return send_from_directory(converted_dir, filename)

@app.route('/save-training-data', methods=['POST'])
def save_training_data():
    user_id = session['user_id']
    try:
        # Generate unique ID
        unique_id = str(uuid.uuid4())
        
        # Get parameters from JSON
        data = request.get_json()
        original_filename = data['original_filename']
        annotations = data['annotations']
        
        # Path setup
        user_upload_dir = os.path.join('users', user_id, 'uploads')
        saved_data_dir = os.path.join('users', user_id, 'saved_data')
        saved_annotations_dir = os.path.join('users', user_id, 'saved_annotations')
        
        # Copy original image
        src_path = os.path.join(user_upload_dir, original_filename)
        dest_filename = f"{unique_id}_{original_filename}"
        dest_path = os.path.join(saved_data_dir, dest_filename)
        shutil.copy2(src_path, dest_path)
        
        # Class mapping
        CLASS_MAP = {
            "SGN": 0,
            "yellow neuron": 1,
            "yellow astrocyte": 2,
            "green neuron": 3,
            "green astrocyte": 4,
            "red neuron": 5,
            "red astrocyte": 6,
            "CD3": 7
        }
        
        # Create YOLO annotations with consistent formatting
        yolo_lines = []
        for ann in annotations:
            class_id = CLASS_MAP.get(ann['class_name'], 0)
            line = "{0} {1:.6f} {2:.6f} {3:.6f} {4:.6f}".format(
                class_id,
                float(ann['x_center']),
                float(ann['y_center']),
                float(ann['width_norm']),
                float(ann['height_norm'])
            )
            yolo_lines.append(line)
        
        # Save annotation file
        os.makedirs(saved_annotations_dir, exist_ok=True)
        with open(os.path.join(saved_annotations_dir, f"{unique_id}.txt"), 'w') as f:
            f.write("\n".join(yolo_lines))
        
        return jsonify({'message': 'Training data saved with clean formatting'})
        
    except Exception as e:
        print(f"Error saving training data: {str(e)}")
        return jsonify({'error': str(e)}), 500
    

@app.route('/clear-training-data', methods=['POST'])
def clear_training_data():
    user_id = session['user_id']
    saved_data_dir = os.path.join('users', user_id, 'saved_data')
    saved_annotations_dir = os.path.join('users', user_id, 'saved_annotations')
    try:
        # Clear saved data
        data_folder = saved_data_dir
        clear_folder(data_folder)
        
        # Clear saved annotations
        annotations_folder = saved_annotations_dir
        clear_folder(annotations_folder)
        
        return jsonify({'message': 'Training data cleared successfully'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/detect-madm', methods=['POST'])
def detect_madm():
    import json

    user_id = session['user_id']
    threshold = float(request.json.get('threshold', 0.5))
    upload_dir = os.path.join('users', user_id, 'uploads')
    tiles_dir = os.path.join('users', user_id, 'images', 'tiles')
    output_txt_dir = os.path.join('users', user_id, 'images', 'tiles_output')
    merged_output_path = os.path.join('users', user_id, 'finaloutput', 'merged_madm.txt')

    os.makedirs(tiles_dir, exist_ok=True)
    os.makedirs(output_txt_dir, exist_ok=True)

    try:
        files = [f for f in os.listdir(upload_dir) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))]
        if not files:
            return jsonify({'error': 'No image found. Upload an image first.'}), 400

        image_path = os.path.join(upload_dir, files[0])
        model_path = 'snapshots/MADM_best.pt'

        # Step 1: Split into tiles
        subprocess.run(['python3', 'scripts/split_image.py', '--image', image_path, '--output', tiles_dir], check=True)

        # Step 2: Detect all tiles using GPU-efficient function
        detect_tiles_in_batch(tiles_dir, output_txt_dir, model_path, threshold)

        # Get image size from original
        with Image.open(image_path) as img:
            image_width, image_height = img.size

        # Step 3: Merge annotations
        subprocess.run([
            'python3', 'scripts/merge_annotations.py',
            '--tiles', output_txt_dir,
            '--output', merged_output_path,
            '--image_width', str(image_width),
            '--image_height', str(image_height)
        ], check=True)

        with open(merged_output_path, 'r') as f:
            final_annotation = f.read()

        return jsonify({
            "annotations": final_annotation,
            "image_width": image_width,
            "image_height": image_height
        })

    except Exception as e:
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500




@app.route('/train-saved', methods=['POST'])
def train_saved_data():
    import shutil
    import time
    import os
    from PIL import Image
    from ultralytics import YOLO
    from scripts.normalization import normalize_image
    import subprocess

    user_id       = session['user_id']
    snapshot_dir  = os.path.join('users', user_id, 'snapshots')
    yolo_base     = os.path.join('users', user_id, 'yolo_dataset')
    img_dir       = os.path.join(yolo_base, 'images')
    lbl_dir       = os.path.join(yolo_base, 'labels')

    try:
        # --- 1. Parse inputs ---
        num_images = int(request.form.get('num_images', '0'))
        model_type = request.form.get('model_type', 'SGN')
        epochs     = int(request.form.get('epochs', '20'))

        print(f"[DEBUG] Inputs → num_images={num_images}, model_type={model_type}, epochs={epochs}")

        # --- 2. Prep dataset dirs ---
        shutil.rmtree(yolo_base, ignore_errors=True)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        

        # --- 3. Copy SAVED DATA (uploaded manually) ---
# --- 3. COPY SAVED DATA (uploaded images + annotations) ---
        print("[DEBUG] Copying saved training data...")

        saved_data_dir = os.path.join('users', user_id, 'saved_data')
        saved_annot_dir = os.path.join('users', user_id, 'saved_annotations')

        saved_imgs = [
            f for f in os.listdir(saved_data_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))
        ]

        print(f"[DEBUG] Found {len(saved_imgs)} saved images")

        for fname in saved_imgs:
            src_img = os.path.join(saved_data_dir, fname)
            dst_img = os.path.join(img_dir, fname)

            # Normalize to RGB
            try:
                if fname.lower().endswith(('.tif', '.tiff')):
                    temp_path = os.path.join(img_dir, f"temp_{fname}.png")
                    normalize_image(src_img, temp_path)
                    with Image.open(temp_path) as im:
                        if im.mode != 'RGB':
                            im = im.convert('RGB')
                        im.save(dst_img)
                    os.remove(temp_path)
                else:
                    with Image.open(src_img) as im:
                        if im.mode != 'RGB':
                            im = im.convert('RGB')
                        im.save(dst_img)
                print(f"[DEBUG] Copied image: {fname}")
            except Exception as e:
                print(f"[ERROR] Failed to copy image {fname}: {e}")
                continue

            # Copy corresponding annotation
            # Get the unique_id from the filename (format is "{unique_id}_{original_name}")
            unique_id = fname.split('_')[0]
            src_lbl = os.path.join(saved_annot_dir, f"{unique_id}.txt")
            dst_lbl = os.path.join(lbl_dir, os.path.splitext(fname)[0] + '.txt')

            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, dst_lbl)
                print(f"[DEBUG] Copied label: {src_lbl} to {dst_lbl}")
            else:
                print(f"[WARNING] Label missing for {fname} (expected {src_lbl})")


        # --- 4. Copy optional pre-train images + labels ---
        # Copy optional pre-train images + labels ---
        pre_dir = 'pre_train_MADM' if model_type == 'MADM' else 'pre_train_SGN' if model_type == 'SGN' else 'pre_train_CD3'
        labels_sub = os.path.join(pre_dir, 'yolo_labels')  # Changed from 'yolo_labels' to 'labels'
        print(f"[DEBUG] Using pre-train dir: {pre_dir}")

        all_imgs = sorted([
            f for f in os.listdir(pre_dir)
            if f.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff'))
        ])
        selected = all_imgs[:num_images]
        print(f"[DEBUG] Copying {len(selected)} pre-train images")

        for fname in selected:
            src_img = os.path.join(pre_dir, fname)
            dst_img = os.path.join(img_dir, fname)

            try:
                if fname.lower().endswith(('.tif', '.tiff')):
                    temp_path = os.path.join(img_dir, f"temp_{fname}.png")
                    normalize_image(src_img, temp_path)
                    with Image.open(temp_path) as im:
                        if im.mode != 'RGB':
                            im = im.convert('RGB')
                        im.save(dst_img)
                    os.remove(temp_path)
                else:
                    with Image.open(src_img) as im:
                        if im.mode != 'RGB':
                            im = im.convert('RGB')
                        im.save(dst_img)
                print(f"[DEBUG] Copied and normalized image {fname}")
            except Exception as e:
                print(f"[ERROR] image copy {fname}: {e}")
                continue  # Skip to next image if current one fails

            # Handle label copying more robustly
            base = os.path.splitext(fname)[0] + '.txt'
            
            # Check multiple possible label locations
            possible_label_locations = [
                os.path.join(labels_sub, base),  # Primary location
                os.path.join(pre_dir, base),     # Alternative location
                os.path.join(pre_dir, 'labels', base)  # Another common location
            ]
            
            label_copied = False
            for src_lbl in possible_label_locations:
                if os.path.exists(src_lbl):
                    dst_lbl = os.path.join(lbl_dir, base)
                    shutil.copy2(src_lbl, dst_lbl)
                    print(f"[DEBUG] Copied label from {src_lbl} to {dst_lbl}")
                    label_copied = True
                    break
            
            if not label_copied:
                print(f"[WARNING] Could not find label for {fname} in any of these locations:")
                for loc in possible_label_locations:
                    print(f"  - {loc}")

        # --- 5. Write data.yaml ---
        class_names = ["SGN"] if model_type == 'SGN' else sorted(
            [os.path.splitext(f)[0] for f in os.listdir(labels_sub)]
        )
        nc = 1 if model_type in ('SGN', 'CD3') else len(class_names)
        yaml_path = os.path.join(yolo_base, 'data.yaml')
        with open(yaml_path, 'w') as f:
            f.write(f"path: {os.path.abspath(yolo_base)}\n")
            f.write("train: images\nval: images\n")
            f.write(f"nc: {nc}\n")
            f.write(f"names: {class_names}\n")

        print(f"[DEBUG] data.yaml written with nc={nc}, names={class_names}")

        # --- 6. Train model ---
        weights = 'snapshots/SGN_best.pt' if model_type == 'SGN' else 'snapshots/CD3_best.pt' if model_type == 'CD3' else 'snapshots/MADM_best.pt'
        run_name = f"run_{int(time.time())}"
        print(f"[DEBUG] Starting YOLO train, weights={weights}, run name={run_name}")
        model = YOLO(weights)
        model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=640,
            batch=4,
            project=snapshot_dir,
            name=run_name,
            save=True
        )

        best = os.path.join(snapshot_dir, run_name, 'weights', 'best.pt')
        time.sleep(3)

        if not os.path.exists(best):
            return jsonify({'error': 'best.pt not found after training'}), 500

        final = os.path.join(snapshot_dir, f"{model_type}_finetuned.pt")
        shutil.copy2(best, final)
        print(f"[DEBUG] Copied final model to {final}")

        # --- 7. Check sample count ---
        valid_samples = [
            f for f in os.listdir(img_dir)
            if os.path.exists(os.path.join(lbl_dir, os.path.splitext(f)[0] + '.txt'))
        ]
        if len(valid_samples) < 5:
            msg = f"Not enough data for K-Fold (need 5, got {len(valid_samples)})"
            print(f"[WARNING] {msg}")
            return jsonify({
                'model_url': f"/{final}",
                'kfold_results': msg
            })

        # --- 8. Run K-Fold ---
        kfold_dir = os.path.join(snapshot_dir, run_name, 'kfold')
        os.makedirs(kfold_dir, exist_ok=True)

        cmd = [
            'python3', 'scripts/kfold_train.py',
            '--image_dir', img_dir,
            '--label_dir', lbl_dir,
            '--weights', best,
            '--epochs', str(epochs),
            '--output_dir', kfold_dir,
            '--nc', str(nc),
            '--names'
        ] + class_names

        print(f"[DEBUG] Running 5-fold validation...")
        subprocess.run(cmd, check=True)

        # --- 9. Read kfold_results.txt ---
        kfold_result_path = os.path.join(kfold_dir, 'kfold_results.txt')
        kfold_text = ""
        if os.path.exists(kfold_result_path):
            with open(kfold_result_path, 'r') as f:
                kfold_text = f.read()

        return jsonify({
            'model_url': f"/snapshots/{model_type}_finetuned.pt",
            'kfold_results': kfold_text
        })

    except Exception as e:
        print(f"[ERROR] /train-saved exception: {e}")
        return jsonify({'error': str(e)}), 500




@app.route('/train', methods=['POST'])
def train_model():
    user_id = session['user_id']
    user_snapshot_dir = os.path.join('users', user_id, 'snapshots')
    user_ft_upload_dir = os.path.join('users', user_id, 'ft_upload')

    try:
        # Clean and setup directories
        shutil.rmtree('ft_upload', ignore_errors=True)
        os.makedirs('ft_upload', exist_ok=True)

        # Save CSV
        csv_file = request.files['csv']
        csv_path = os.path.join(user_ft_upload_dir, 'annotations.csv')  # Define csv_path
        csv_file.save(csv_path)

        # Save images
        for img in request.files.getlist('images'):
            img.save(os.path.join(user_ft_upload_dir, img.filename))

        # Get training parameters
        model_type = request.form.get('model_type', 'SGN')
        epochs = request.form.get('epochs', '10')
        classes_file = 'monochrome.csv' if model_type == 'SGN' else 'color.csv'  # Define classes_file
        weights_file = 'snapshots/SGN_Rene.h5' if model_type == 'SGN' else 'snapshots/MADMweights.h5'

        # Validate epochs
        try:
            epochs = int(epochs)
            if epochs < 1:
                return jsonify({'error': 'Epochs must be at least 1'}), 400
        except ValueError:
            return jsonify({'error': 'Invalid epochs value'}), 400

        tb_log_dir = os.path.join(user_snapshot_dir, 'tensorboard')
        os.makedirs(tb_log_dir, exist_ok=True)
        # Run training
        cmd = [
            'python3', 'keras_retinanet/keras_retinanet/bin/train.py',
            '--tensorboard-dir', tb_log_dir,
            '--weights', weights_file,
            '--lr', '1e-4',
            '--batch-size', '8',
            '--epochs', str(epochs),
            '--snapshot-path', user_snapshot_dir,  # ✅ User-specific snapshots
            'csv', 
            csv_path,  # Now defined
            classes_file  # Now defined
        ]

        # Run with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        # Stream output to terminal
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip(), flush=True)

        if process.returncode != 0:
            return jsonify({'error': 'Training failed'}), 500

        # Return latest snapshot
        # Get specific snapshot based on epochs
        epoch_str = f"{epochs:02d}"
        expected_filename = f'resnet50_csv_{epoch_str}.h5'
        snapshot_path = os.path.join(user_snapshot_dir, expected_filename)
    
            # ===== START CRITICAL FIX =====
        # Wait for file to finish writing
        time.sleep(1)  # Wait 1 second
        gc.collect()  # Clean up memory

        # Copy using safe binary method
        fixed_path = os.path.join(user_snapshot_dir, 'last_used.h5')
        with open(snapshot_path, 'rb') as src_file, open(fixed_path, 'wb') as dest_file:
            shutil.copyfileobj(src_file, dest_file)
        # ===== END CRITICAL FIX =====

        return send_file(
            snapshot_path,
            as_attachment=True,
            download_name=expected_filename,
            mimetype='application/octet-stream'
        )

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/detect-custom', methods=['POST'])
def detect_custom():
    user_id = session['user_id']
    try:
        # Get uploaded model and image
        pt_file = request.files['pt_file']  # Changed from h5_file to pt_file
        model_type = request.form.get('model_type', 'SGN')
        threshold = float(request.form.get('threshold', 0.5))  # Add threshold parameter
        
        upload_dir = os.path.join('users', user_id, 'uploads')
        final_output = os.path.join('users', user_id, 'finaloutput')
        
        # Save model temporarily
        model_path = os.path.join(upload_dir, secure_filename(pt_file.filename))
        pt_file.save(model_path)

        # Find uploaded image
        image_files = [f for f in os.listdir(upload_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]
        if not image_files:
            return jsonify({'error': 'No image found'}), 400
            
        image_name = image_files[0]
        image_path = os.path.join(upload_dir, image_name)

        # Load image
        with Image.open(image_path) as img:
            orig_width, orig_height = img.size
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                if img.mode in ('I;16', 'I;16B', 'I;16L'):
                    img_array = np.array(img)
                    min_val = np.min(img_array)
                    max_val = np.max(img_array)
                    if max_val > min_val:
                        normalized = ((img_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                    else:
                        normalized = (img_array // 256).astype(np.uint8)
                    img = Image.fromarray(normalized)
                img = img.convert('RGB')

        # Load YOLO model
        model = YOLO(model_path)
        
        # Run prediction
        results = model.predict(
            source=img,
            conf=threshold,
            save=False,
            save_txt=False,
            save_conf=True
        )
        
        # Define class names based on model type
        if model_type == 'SGN':
            CLASS_NAMES = ["SGN"]
        elif model_type == 'CD3':
            CLASS_NAMES = ["CD3"]
        else:  # MADM
            CLASS_NAMES = [
                "SGN",
                "yellow neuron",
                "yellow astrocyte",
                "green neuron",
                "green astrocyte",
                "red neuron",
                "red astrocyte"
            ]
        # Process results
        detections = []
        for result in results:
            for box in result.boxes:
                # Get normalized coordinates
                x_center, y_center, width, height = box.xywhn[0].cpu().numpy()
                conf = box.conf.item()
                cls = int(box.cls.item())
                
                # Convert to absolute coordinates
                abs_x_center = x_center * orig_width
                abs_y_center = y_center * orig_height
                abs_width = width * orig_width
                abs_height = height * orig_height
                
                # Convert to top-left coordinates
                x1 = max(0, abs_x_center - (abs_width / 2))
                y1 = max(0, abs_y_center - (abs_height / 2))
                x2 = min(orig_width, x1 + abs_width)
                y2 = min(orig_height, y1 + abs_height)
                
                # Map class index to name
                class_name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"unknown_{cls}"
                
                detections.append({
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'class': class_name
                })
        
        # Format as CSV
        csv_lines = []
        for detection in detections:
            csv_line = f"{image_name},{detection['x1']},{detection['y1']},{detection['x2']},{detection['y2']},{detection['class']}"
            csv_lines.append(csv_line)
        
        csv_data = "\n".join(csv_lines)
        
        return jsonify({'annotations': csv_data})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Cleanup model file
        try:
            os.remove(model_path)
        except:
            pass

@app.route('/scale-image', methods=['POST'])
def scale_image():
    user_id = session['user_id']
    upload_dir = os.path.join('users', user_id, 'uploads')
    converted_dir = os.path.join('users', user_id, 'converted')
    try:
        diameter = float(request.form['diameter'])
        original_filename = request.form['original_filename']
        
        if 'target_diameter' not in session:
            session['target_diameter'] = 34.0

        scaling_factor = session['target_diameter'] / diameter
        session['target_diameter'] = diameter

        current_path = os.path.join(upload_dir, original_filename)
        if not os.path.exists(current_path):
            return jsonify({'error': 'Current image not found'}), 400

        with Image.open(current_path) as img:
            new_width = int(img.width * scaling_factor)
            new_height = int(img.height * scaling_factor)
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save resized image
            resized_path = os.path.join(upload_dir, original_filename)
            resized_img.save(resized_path, format='TIFF', compression='tiff_deflate')
            
            # Create normalized preview
            unique_id = str(uuid.uuid4())
            output_filename = f"{unique_id}.png"
            output_path = os.path.join(converted_dir, output_filename)
            normalize_image(resized_path, output_path)

            return jsonify({
                'converted_url': f'/converted/{output_filename}',
                'scaling_factor': scaling_factor,
                'new_width': new_width,
                'new_height': new_height
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/detect-finetuned', methods=['POST'])
def detect_finetuned():
    user_id = session['user_id']
    user_cleanup_dirs = [
            os.path.join('users', user_id, 'output'),
            os.path.join('users', user_id, 'input'),
            os.path.join('users', user_id, 'images')
        ]
    try:
        # Get model type from request
        model_type = request.json.get('model_type', 'SGN')
        # 1. Get the copied model
        user_snapshot_dir = os.path.join('users', user_id, 'snapshots')
        model_path = os.path.join(user_snapshot_dir, 'last_used.h5')  # ✅ User's model
        
        # 2. Basic validation
        if not os.path.exists(model_path):
            return jsonify({'error': 'No trained model found'}), 400

        # 3. Find uploaded image
        upload_dir = os.path.join('users', user_id, 'uploads')
        output_dir = os.path.join('users', user_id, 'output')
        output_csv_dir = os.path.join('users', user_id, 'output', 'output_csv')
        image_files = [f for f in os.listdir(upload_dir) if f.endswith(('.tiff', '.tif'))]
        if not image_files:
            return jsonify({'error': 'No image uploaded'}), 400
        image_path = os.path.join(upload_dir, image_files[0])

        # 4. Use same CSV path as custom detection
        csv_filename = f"{os.path.basename(image_path)}_result.csv"
        csv_path = os.path.join(output_csv_dir, csv_filename)
        detection_script = 'scripts/custom_detection_color.py' if model_type == 'MADM' else 'scripts/custom_detection.py'

        subprocess.run([
            'python3',
            detection_script,
            image_path,
            model_path,
            output_dir
        ], check=True)




        # 6. Read and return results
        with open(csv_path, 'r') as f:
            csv_data = f.read()

        #clear directories to prevent buildup of old data:

        for dir_name in user_cleanup_dirs:
            dir_path = os.path.join(os.getcwd(), dir_name)
            if os.path.exists(dir_path):
                clear_folder(dir_path)

        return jsonify({'annotations': csv_data})
        

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def batch_process_image(user_id, image_path, detection_type, threshold, model_path=None,class_type='SGN', cell_diameter=34):
    """Process single image using existing pipeline"""
    try:
        # Get user directories
        upload_dir = os.path.join('users', user_id, 'uploads')
        input_dir = os.path.join('users', user_id, 'input')
        images_dir = os.path.join('users', user_id, 'images')
        output_dir = os.path.join('users', user_id, 'output')
        final_dir = os.path.join('users', user_id, 'finaloutput')

        # Clear directories before processing
        for dir_path in [upload_dir, input_dir, images_dir, output_dir, final_dir]:
            clear_folder(dir_path)
            os.makedirs(dir_path, exist_ok=True)

        # Copy image to uploads
        shutil.copy(image_path, os.path.join(upload_dir, os.path.basename(image_path)))

        # Scale image
        with Image.open(image_path) as img:
            scaling_factor = 34.0 / cell_diameter
            new_width = int(img.width * scaling_factor)
            new_height = int(img.height * scaling_factor)
            scaled_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            scaled_img.save(image_path, format='TIFF', compression='tiff_deflate')


        if detection_type == 'SGN':
            scripts = [
                ['python3', 'scripts/8to16bit.py', upload_dir, input_dir],
                ['python3', 'scripts/splitimage.py', input_dir, images_dir],
                ['python3', 'scripts/detection_SGN.py', images_dir, output_dir, str(threshold)],
                ['python3', 'scripts/mergecsv.py', os.path.join(output_dir, 'output_csv'), 
                 os.path.join(final_dir, 'annotations.csv')]
            ]
        elif detection_type == 'MADM':
            scripts = [
                ['python3', 'scripts/8to16bit.py', upload_dir, input_dir],
                ['python3', 'scripts/splitimage.py', input_dir, images_dir],
                ['python3', 'scripts/detection.py', images_dir, output_dir, str(threshold)],
                ['python3', 'scripts/mergecsv.py', os.path.join(output_dir, 'output_csv'),
                 os.path.join(final_dir, 'annotations.csv')]
            ]
        else:  # Custom
            # Determine which script to use based on class type
            if class_type == 'SGN':
                script_name = 'scripts/batch_SGN_custom.py'
            else:
                script_name = 'scripts/batch_MADM_custom.py'
                
            scripts = [
                ['python3', 'scripts/8to16bit.py', upload_dir, input_dir],
                ['python3', 'scripts/splitimage.py', input_dir, images_dir],
                ['python3', script_name, images_dir, output_dir, str(threshold), model_path],
                ['python3', 'scripts/mergecsv.py', os.path.join(output_dir, 'output_csv'), 
                os.path.join(final_dir, 'annotations.csv')]
            ]

        # Execute scripts
        for script in scripts:
            result = subprocess.run(script, capture_output=True, text=True)
            if result.returncode != 0:
                return {'success': False, 'error': result.stderr}

        # Read results
        csv_path = os.path.join(final_dir, 'annotations.csv')
        with open(csv_path, 'r') as f:
            return {'success': True, 'csv': f.read(), 'filename': os.path.basename(image_path)}

    except Exception as e:
        return {'success': False, 'error': str(e)}

@app.route('/batch-detect', methods=['POST'])
def batch_detect():
    user_id = session['user_id']
    try:
        # Create batch directory
        batch_dir = os.path.join('users', user_id, 'batch_temp')
        os.makedirs(batch_dir, exist_ok=True)
        clear_folder(batch_dir)  # Clear previous temp files

        # Save uploaded files
        for file in request.files.getlist('images'):
            file.save(os.path.join(batch_dir, secure_filename(file.filename)))

        # Get parameters
        detection_type = request.form['detection_type']
        threshold = float(request.form.get('threshold', 0.5))
        custom_model = request.files.get('custom_model')
        class_type = request.form.get('class_type', 'SGN')
        cell_diameter = float(request.form.get('cell_diameter', 34))

        # Handle custom model
        model_path = None
        if detection_type == 'custom' and custom_model:
            model_path = os.path.join(batch_dir, 'custom_model.h5')
            custom_model.save(model_path)

        # Process each image
        results = []
        for filename in os.listdir(batch_dir):
            file_path = os.path.join(batch_dir, filename)
            if os.path.isfile(file_path) and filename.lower().endswith(('.tiff', '.tif')):
                result = batch_process_image(
                    user_id=user_id,
                    image_path=file_path,
                    detection_type=detection_type,
                    threshold=threshold,
                    model_path=model_path,
                    class_type=class_type,
                    cell_diameter=cell_diameter
                )
                if result['success']:
                    results.append({
                        'csv': result['csv'],
                        'image': filename,
                        'csv_name': f"{os.path.splitext(filename)[0]}_annotations.csv"
                    })

        # Create ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for result in results:
                # Add CSV
                zipf.writestr(result['csv_name'], result['csv'])
                # Add original image
                zipf.write(os.path.join(batch_dir, result['image']), result['image'])

        # Cleanup
        shutil.rmtree(batch_dir, ignore_errors=True)
        zip_buffer.seek(0)

        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name='batch_results.zip'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    


@app.route('/snapshots/<path:filename>')
def serve_snapshot(filename):
    user_id = session['user_id']
    snapshot_dir = os.path.join('users', user_id, 'snapshots')
    return send_from_directory(snapshot_dir, filename)


from flask import jsonify, session
from tensorflow.python.summary.summary_iterator import summary_iterator
import os
import glob

@app.route('/events-data', methods=['GET'])
def events_data():
    user_id = session.get('user_id')
    log_dir = os.path.join('users', user_id, 'snapshots', 'tensorboard', 'train')

    if not os.path.exists(log_dir):
        return jsonify({'error': f'Log dir not found: {log_dir}'}), 404

    # Find all event files directly in train/
    event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
    # Only pick files older than 5 seconds (to avoid reading during flush)
    now = time.time()
    event_files = [f for f in event_files if now - os.path.getmtime(f) > 5]
    if not event_files:
        return jsonify({'error': 'No event files found in train/'}), 404

    # Pick the newest one
    event_files.sort(key=os.path.getmtime, reverse=True)
    latest = event_files[0]
    print(f"[events-data] ✅ Reading from: {latest}")

    scalars = {}
    for e in summary_iterator(latest):
        if not e.summary:
            continue
        for v in e.summary.value:
            val = get_scalar_value(v)
            if val is not None:
                print(f"✅ Tag: {v.tag}, value: {val}")
                scalars.setdefault(v.tag, []).append({
                    'step': e.step,
                    'wall_time': e.wall_time,
                    'value': val
                })


    if not scalars:
        return jsonify({'error': 'No scalar values found'}), 404

    return jsonify(scalars)




def delete_expired_sessions():
    now = datetime.datetime.utcnow()  # Use UTC time
    users_dir = 'users'
    for user_id in os.listdir(users_dir):
        user_path = os.path.join(users_dir, user_id)
        if os.path.isdir(user_path):
            try:
                mod_time = datetime.datetime.utcfromtimestamp(os.path.getmtime(user_path))
                if (now - mod_time).total_seconds() > 86400:
                    shutil.rmtree(user_path)
                    print(f"Cleaned expired session: {user_id}")
            except Exception as e:
                print(f"Error cleaning {user_id}: {str(e)}")
# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(func=delete_expired_sessions, trigger="interval", hours=24)
scheduler.start()

# Shut down the scheduler when exiting the app
atexit.register(lambda: scheduler.shutdown())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)