import argparse, os, os.path as osp, numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
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
    return filters.sobel(gray)

def get_saturation_mask(image_rgb, threshold=40):
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    return (saturation > threshold).astype(np.uint8)

def furthest_point_sampling(pixels, coords, k=20, rgb_threshold=30):
    coords = np.array(coords)
    if len(pixels) == 0 or len(pixels) < k or k == 0:
        return coords.tolist()[:k]
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
    return [tuple(c) for c in selected_coords]

def random_hint_selection(coords, k=20):
    coords = np.array(coords)
    if len(coords) == 0 or k == 0:
        return []
    idx = np.random.choice(len(coords), min(k, len(coords)), replace=False)
    return [tuple(coords[i]) for i in idx]

def save_hints(hint_path, selected_coords):
    with open(hint_path, 'w') as f:
        for y, x in selected_coords:
            f.write(f"{x} {y}\n")

def save_debug(debug_img, selected_coords, out_path, r=4):
    draw = ImageDraw.Draw(debug_img)
    for y, x in selected_coords:
        draw.ellipse([(x - r, y - r), (x + r, y + r)], outline='red', width=1)
    debug_img.save(out_path)

def process_image(image_np, mask_np, method, num_hint, args):
    assert mask_np is not None, "Mask is required for all methods in this pipeline!"
    fg_mask = (mask_np > 0)
    img_proc = image_np.copy()
    coords = np.argwhere(fg_mask)
    pixels = img_proc[fg_mask]

    if num_hint == 0:
        return []

    # Stepwise methods
    if method == "random":
        selected_coords = random_hint_selection(coords, k=num_hint)
    elif method == "saturation":
        sat_mask = get_saturation_mask(img_proc, threshold=40)
        valid_mask = fg_mask & (sat_mask > 0)
        coords2 = np.argwhere(valid_mask)
        selected_coords = random_hint_selection(coords2, k=num_hint)
    elif method == "clahe_saturation":
        img_proc = apply_clahe(img_proc)
        sat_mask = get_saturation_mask(img_proc, threshold=40)
        valid_mask = fg_mask & (sat_mask > 0)
        coords2 = np.argwhere(valid_mask)
        selected_coords = random_hint_selection(coords2, k=num_hint)
    elif method == "saliency":
        saliency = compute_saliency_map(img_proc)
        sal_vals = saliency[fg_mask]
        if len(sal_vals) > 0:
            threshold = np.percentile(sal_vals, 25)
            sal_mask = (saliency >= threshold)
            valid_mask = fg_mask & sal_mask
            coords2 = np.argwhere(valid_mask)
        else:
            coords2 = coords
        selected_coords = random_hint_selection(coords2, k=num_hint)
    elif method == "fps":
        selected_coords = furthest_point_sampling(pixels, coords, k=num_hint)
    elif method == "full_pipeline":
        img_proc = apply_clahe(img_proc)
        sat_mask = get_saturation_mask(img_proc, threshold=40)
        saliency = compute_saliency_map(img_proc)
        valid_mask = fg_mask & (sat_mask > 0)
        sal_vals = saliency[valid_mask]
        coords2 = np.argwhere(valid_mask)
        pixels2 = img_proc[valid_mask]
        if len(sal_vals) > 0:
            threshold = np.percentile(sal_vals, 25)
            sal_mask = (saliency >= threshold)
            valid_mask2 = valid_mask & sal_mask
            coords2 = np.argwhere(valid_mask2)
            pixels2 = img_proc[valid_mask2]
        selected_coords = furthest_point_sampling(pixels2, coords2, k=num_hint)
    else:
        raise ValueError("Unknown hint method: " + method)
    return selected_coords

def make_ablation_hints(args):
    np.random.seed(args.seed)
    os.makedirs(args.hint_dir, exist_ok=True)
    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(args.img_dir) if f.lower().endswith('.jpg')])
    hint_counts = [0, 1, 2, 5, 10, 20]

    for method in args.methods:
        print(f"\nGenerating hints with method: {method}")
        for n_hint in hint_counts:
            hint_subdir = osp.join(args.hint_dir, method, f"h4-n{n_hint}")
            os.makedirs(hint_subdir, exist_ok=True)
            debug_subdir = osp.join(args.debug_dir, method, f"h4-n{n_hint}") if args.debug_dir else None
            if debug_subdir:
                os.makedirs(debug_subdir, exist_ok=True)

            for idx, filename in enumerate(tqdm(image_files, desc=f"{method}-n{n_hint}")):
                img_path = osp.join(args.img_dir, filename)
                mask_filename = osp.splitext(filename)[0] + ".png"
                mask_path = osp.join(args.mask_dir, mask_filename)

                image = Image.open(img_path).convert('RGB').resize((224, 224), Image.BICUBIC)
                mask = Image.open(mask_path).convert('L').resize((224, 224), Image.NEAREST)
                image_np = np.array(image)
                mask_np = np.array(mask)

                selected_coords = process_image(image_np, mask_np, method, n_hint, args)
                renamed_base = f"egypt_{idx+1}"
                hint_txt_path = osp.join(hint_subdir, f"{renamed_base}.txt")
                save_hints(hint_txt_path, selected_coords)

                if debug_subdir:
                    debug_img = image.copy()
                    save_debug(debug_img, selected_coords, osp.join(debug_subdir, f"{renamed_base}.jpg"), r=args.hint_size)
    print("\nAblation hint generation complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate mask-aware ablation study hints for colorization")
    parser.add_argument('--img_dir', type=str, required=True, help='RGB image directory')
    parser.add_argument('--mask_dir', type=str, required=True, help='Binary mask image directory')
    parser.add_argument('--hint_dir', type=str, required=True, help='Output directory for hints (per method/hint count)')
    parser.add_argument('--debug_dir', type=str, default=None, help='Directory for saving visual hint maps')
    parser.add_argument('--hint_size', type=int, default=4, help='Radius size for hint points (debug only)')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed for reproducibility')
    parser.add_argument('--methods', nargs='+',
                        default=['random', 'saturation', 'clahe_saturation', 'saliency', 'fps', 'full_pipeline'],
                        help='Hint selection methods to use')
    args = parser.parse_args()
    make_ablation_hints(args)
