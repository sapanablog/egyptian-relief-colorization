import argparse, os, os.path as osp, numpy as np, random
from PIL import Image, ImageDraw
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import color, filters, exposure
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter


def apply_clahe(image_rgb):
    from cv2 import cvtColor, COLOR_RGB2LAB, COLOR_LAB2RGB, createCLAHE, merge, split
    lab = cvtColor(image_rgb, COLOR_RGB2LAB)
    l, a, b = split(lab)
    clahe = createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    return cvtColor(merge((l_clahe, a, b)), COLOR_LAB2RGB)


def compute_saliency_map(image_rgb):
    gray = color.rgb2gray(image_rgb)
    blurred = gaussian_filter(gray, sigma=3)
    log_map = np.log1p(np.abs(gray - blurred))
    log_map = (log_map - log_map.min()) / (log_map.max() - log_map.min() + 1e-8)
    return log_map


def furthest_point_sampling(pixels, coords, saliency_map, k=20, rgb_threshold=30):
    if len(pixels) < k:
        return coords.tolist()

    saliency_vals = saliency_map[coords[:, 0], coords[:, 1]]
    start_idx = np.argmax(saliency_vals)
    selected_coords = [coords[start_idx]]
    selected_colors = [pixels[start_idx]]

    while len(selected_coords) < k:
        dists = cdist(pixels, np.array(selected_colors))
        min_dist = np.min(dists, axis=1)

        score = min_dist * saliency_vals
        next_idx = np.argmax(score)

        if np.all(cdist([pixels[next_idx]], selected_colors) > rgb_threshold):
            selected_coords.append(coords[next_idx])
            selected_colors.append(pixels[next_idx])

        if len(selected_coords) >= len(pixels):
            break

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
            mask_path = osp.join(args.mask_dir, osp.splitext(filename)[0] + ".png")

            image = Image.open(img_path).convert('RGB').resize((224, 224), Image.BICUBIC)
            mask = Image.open(mask_path).convert('L').resize((224, 224), Image.NEAREST)

            image_np = np.array(image)
            mask_np = np.array(mask)

            image_clahe = apply_clahe(image_np)
            saliency = compute_saliency_map(image_clahe)

            hsv = color.rgb2hsv(image_clahe / 255.0)
            saturation = hsv[:, :, 1]
            mask_np[saturation < 0.15] = 0  # Filter low saturation

            coords = np.argwhere(mask_np > 0)
            pixels = image_clahe[mask_np > 0]

            renamed_base = f"egypt_{idx+1}"
            hint_txt_path = osp.join(hint_subdir, f"{renamed_base}.txt")
            selected_coords = []

            if coords.shape[0] > 0 and num_hint > 0:
                selected_coords = furthest_point_sampling(pixels, coords, saliency, k=num_hint)

            with open(hint_txt_path, 'w') as f:
                for y, x in selected_coords:
                    f.write(f"{x} {y}\n")

            if args.debug_dir:
                debug_img = Image.fromarray(image_clahe.copy())
                draw = ImageDraw.Draw(debug_img)
                for y, x in selected_coords:
                    r = args.hint_size
                    draw.ellipse([(x - r, y - r), (x + r, y + r)], outline='red', width=1)
                debug_img.save(osp.join(args.debug_dir, f"{num_hint}_hints", f"{renamed_base}.jpg"))

    print("Hint generation complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hint generation using CLAHE + Saturation + Fallback Saliency + Furthest Sampling")
    parser.add_argument('--img_dir', type=str, required=True, help='RGB image folder')
    parser.add_argument('--mask_dir', type=str, required=True, help='Mask image folder (.png)')
    parser.add_argument('--hint_dir', type=str, required=True, help='Output .txt hint folder')
    parser.add_argument('--debug_dir', type=str, default=None, help='Optional overlay image output')
    parser.add_argument('--hint_size', type=int, default=2, help='Hint circle radius')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    args = parser.parse_args()

    make_mask_guided_hints(args)
