# merge_annotations.py

import os
import argparse

def merge_annotations(tiles_dir, output_file, tile_size=512, image_width=None, image_height=None):
    with open(output_file, 'w') as out:
        for file in os.listdir(tiles_dir):
            if file.endswith('.txt') and file.startswith("tile_"):
                # Parse coordinates from filename
                parts = file.replace(".txt", "").split('_')
                if len(parts) != 3:
                    continue
                x_offset = int(parts[1])
                y_offset = int(parts[2])

                with open(os.path.join(tiles_dir, file)) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        cls, cx, cy, w, h = map(float, parts)

                        # Convert back to full image coords
                        cx = cx * tile_size + x_offset
                        cy = cy * tile_size + y_offset
                        w = w * tile_size
                        h = h * tile_size

                        # Convert back to normalized full-image coords if desired
                        if image_width and image_height:
                            cx /= image_width
                            cy /= image_height
                            w /= image_width
                            h /= image_height

                        out.write(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    print(f"Saved merged annotations to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiles', required=True, help='Directory containing YOLO .txt files for tiles')
    parser.add_argument('--output', required=True, help='Path to output merged annotation file')
    parser.add_argument('--image_width', type=int, help='Original image width (optional)')
    parser.add_argument('--image_height', type=int, help='Original image height (optional)')
    args = parser.parse_args()

    merge_annotations(args.tiles, args.output, image_width=args.image_width, image_height=args.image_height)
