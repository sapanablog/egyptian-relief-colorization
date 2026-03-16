import argparse, os, os.path as osp, numpy as np, random
from PIL import Image, ImageDraw
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import cv2

def apply_clahe(image_rgb):
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)

def compute_saliency_score(image_np, mask_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    contrast_score = np.abs(laplacian)
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    score = contrast_score * (saturation / 255.0)
    score *= (mask_np > 0)
    return score

def furthest_point_sampling_with_saliency(pixels, coords, score_map, k=20, rgb_threshold=30):
    if len(pixels) < k:
        return coords.tolist()

    flat_scores = score_map[coords[:, 0], coords[:, 1]]
    sorted_indices = np.argsort(flat_scores)[::-1]
    coords = coords[sorted_indices]
    pixels = pixels[sorted_indices]

    selected_coords = [coords[0]]
    selected_colors = [pixels[0]]

    for i in range(1, len(coords)):
        if len(selected_coords) >= k:
            break
        color = pixels[i]
        dist = np.min(cdist([color], selected_colors))
        if dist > rgb_threshold:
            selected_coords.append(coords[i])
            selected_colors.append(color)

    return [(y, x) for y, x in selected_coords]

def make_mask_guided_hints(args):
    os.makedirs(args.hint_dir, exist_ok=True)
    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(args.img_dir) if f.lower().endswith('.jpg')])
    random.seed(args.seed)

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

            image_np = np.array(apply_clahe(np.array(image)))
            mask_np = np.array(mask)

            coords = np.argwhere(mask_np > 0)
            pixels = image_np[mask_np > 0]

            renamed_base = f"egypt_{idx+1}"
            hint_txt_path = osp.join(hint_subdir, f"{renamed_base}.txt")
            selected_coords = []

            if coords.shape[0] > 0 and num_hint > 0:
                score_map = compute_saliency_score(image_np, mask_np)
                selected_coords = furthest_point_sampling_with_saliency(pixels, coords, score_map, k=num_hint)

            with open(hint_txt_path, 'w') as f:
                for y, x in selected_coords:
                    f.write(f"{x} {y}\\n")

            if args.debug_dir:
                debug_img = Image.fromarray(image_np.copy())
                draw = ImageDraw.Draw(debug_img)
                for y, x in selected_coords:
                    r = args.hint_size
                    draw.ellipse([(x - r, y - r), (x + r, y + r)], outline='red', width=1)
                debug_img.save(osp.join(args.debug_dir, f"{num_hint}_hints", f"{renamed_base}.jpg"))

    print("Hint generation complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Enhanced Hint Generator using CLAHE + Saliency + Furthest Sampling")
    parser.add_argument('--img_dir', type=str, required=True, help='Directory of RGB images')
    parser.add_argument('--mask_dir', type=str, required=True, help='Directory of binary mask images (.png)')
    parser.add_argument('--hint_dir', type=str, required=True, help='Output directory for hint .txt files')
    parser.add_argument('--debug_dir', type=str, default=None, help='Directory to save visual debug overlays')
    parser.add_argument('--hint_size', type=int, default=2, help='Radius size of each hint point')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    args = parser.parse_args()
    make_mask_guided_hints(args)