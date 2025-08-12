# normalization.py - for frontend
import numpy as np
from PIL import Image

def percentile_normalization(img_array, low_percentile=1, high_percentile=99):
    """Normalize image using percentile-based scaling"""
    p_low = np.percentile(img_array, low_percentile)
    p_high = np.percentile(img_array, high_percentile)
    
    # Clip and scale to [0,255], convert to uint8
    arr_clipped = np.clip(img_array, p_low, p_high)
    arr_normalized = ((arr_clipped - p_low) * 255.0 / max(1, (p_high - p_low))).astype(np.uint8)
    return arr_normalized

def normalize_image(input_path, output_path):
    """Normalize an image file and save the result"""
    with Image.open(input_path) as img:
        if img.mode == 'I;16':
            img_array = np.array(img).astype(np.uint16)
            normalized = percentile_normalization(img_array)
            result = Image.fromarray(normalized)
        else:
            # For non-16bit images, just convert to RGB if needed
            if img.mode != 'RGB':
                result = img.convert('RGB')
            else:
                result = img.copy()
        
        result.save(output_path)