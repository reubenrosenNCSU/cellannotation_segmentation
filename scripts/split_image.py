# split_image.py

import os
import argparse
from PIL import Image

def split_image(image_path, output_dir, tile_size=512):
    os.makedirs(output_dir, exist_ok=True)
    img = Image.open(image_path)
    width, height = img.size

    tile_id = 0
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            box = (x, y, min(x + tile_size, width), min(y + tile_size, height))
            tile = img.crop(box)
            
            filename = f"tile_{x}_{y}.png"
            tile.save(os.path.join(output_dir, filename))
            tile_id += 1

    print(f"Split into {tile_id} tiles at {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--output', required=True, help='Directory to save tiles')
    args = parser.parse_args()

    split_image(args.image, args.output)
