import argparse, os, os.path as osp, numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import filters
from scipy.spatial.distance import cdist
import cv2

def apply_clahe(image_rgb):
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)

def compute_saliency_map(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    saliency = filters.sobel(gray)
    return saliency

def furthest_point_sampling(pixels, coords, k=20, rgb_threshold=30):
    coords = np.array(coords)
    if len(pixels) < k:
        return coords.tolist()

    sort_idx = np.lexsort((coords[:, 1], coords[:, 0]))
    coords = coords[sort_idx]
    pixels = pixels[sort_idx]

    selected_coords = [coords[0]]
    selected_colors = [pixels[0]]
    remaining = list(range(1, len(pixels)))

    while len(selected_coords) < k and remaining:
        dists = cdist(pixels[remaining], np.array(selected_colors))
        min_dist = np.min(dists, axis=1)
        farthest_idx = remaining[np.argmax(min_dist)]

        color = pixels[farthest_idx]
        if np.all(cdist([color], selected_colors) > rgb_threshold):
            selected_coords.append(coords[farthest_idx])
            selected_colors.append(color)

        remaining.remove(farthest_idx)

    return [(y, x) for y, x in selected_coords]

def make_mask_guided_hints(args):
    os.makedirs(args.hint_dir, exist_ok=True)
    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(args.img_dir) if f.lower().endswith('.jpg')])

    for num_hint in [0, 1, 2, 5, 10, 20]:
        hint_subdir = osp.join(args.hint_dir, f"h{args.hint_size}-n{num_hint}")
        os.makedirs(hint_subdir, exist_ok=True)
        if args.debug_dir:
            os.makedirs(osp.join(args.debug_dir, f"{num_hint}_hints"), exist_ok=True)

        for idx, filename in enumerate(tqdm(image_files, desc=f"Generating {num_hint} hints")):
            img_path = osp.join(args.img_dir, filename)
            mask_filename = osp.splitext(filename)[0] + ".png"
            mask_path = osp.join(args.mask_dir, mask_filename)

            image = Image.open(img_path).convert('RGB').resize((224, 224), Image.BICUBIC)
            mask = Image.open(mask_path).convert('L').resize((224, 224), Image.NEAREST)

            image_np = np.array(image)
            mask_np = np.array(mask)

            image_clahe = apply_clahe(image_np.copy())
            saliency = compute_saliency_map(image_clahe)

            hsv = cv2.cvtColor(image_clahe, cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1]
            sat_mask = saturation > 60

            combined_mask = np.logical_and(mask_np > 0, sat_mask)
            coords = np.argwhere(combined_mask)
            pixels = image_clahe[combined_mask]

            saliency_vals = saliency[combined_mask]
            if len(saliency_vals) == 0:
                selected_coords = []
            else:
                threshold = np.percentile(saliency_vals, 50)
                valid_idx = np.where(saliency_vals >= threshold)[0]
                if len(valid_idx) == 0:
                    selected_coords = []
                else:
                    coords = coords[valid_idx]
                    pixels = pixels[valid_idx]
                    selected_coords = furthest_point_sampling(pixels, coords, k=num_hint)

            renamed_base = f"egypt_{idx+1}"
            hint_txt_path = osp.join(hint_subdir, f"{renamed_base}.txt")
            with open(hint_txt_path, 'w') as f:
                for y, x in selected_coords:
                    f.write(f"{x} {y}\n")

            if args.debug_dir:
                debug_img = Image.fromarray(image_np.copy())
                draw = ImageDraw.Draw(debug_img)
                for y, x in selected_coords:
                    r = args.hint_size
                    draw.ellipse([(x - r, y - r), (x + r, y + r)], outline='red', width=1)
                debug_img.save(osp.join(args.debug_dir, f"{num_hint}_hints", f"{renamed_base}.jpg"))

    print("Hint generation complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate structured hints using CLAHE + Saliency + FPS")
    parser.add_argument('--img_dir', type=str, required=True, help='RGB image directory')
    parser.add_argument('--mask_dir', type=str, required=True, help='Binary mask image directory')
    parser.add_argument('--hint_dir', type=str, required=True, help='Output directory for hints')
    parser.add_argument('--debug_dir', type=str, default=None, help='Directory for saving visual hint maps')
    parser.add_argument('--hint_size', type=int, default=4, help='Radius size for hint points')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed for reproducibility')
    args = parser.parse_args()

    make_mask_guided_hints(args)
