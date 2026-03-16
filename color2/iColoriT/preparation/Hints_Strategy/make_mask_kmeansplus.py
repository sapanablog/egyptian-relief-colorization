import argparse, os, os.path as osp, numpy as np, random
from PIL import Image, ImageDraw
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def apply_clahe(image_rgb):
    import cv2
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)

def furthest_point_sampling(pixels, coords, k=20, rgb_threshold=30):
    selected_coords = []
    if len(pixels) < k:
        return coords.tolist()

    selected_coords.append(coords[np.random.randint(len(pixels))])
    selected_colors = [pixels[coords.tolist().index(selected_coords[0])]]

    while len(selected_coords) < k:
        dists = cdist(pixels, np.array(selected_colors))
        min_dist = np.min(dists, axis=1)
        farthest_idx = np.argmax(min_dist)

        color = pixels[farthest_idx]
        if np.all(cdist([color], selected_colors) > rgb_threshold):
            selected_coords.append(coords[farthest_idx])
            selected_colors.append(color)

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
                selected_coords = furthest_point_sampling(pixels, coords, k=num_hint)

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
    parser = argparse.ArgumentParser(description="Generate high-quality hints for iColoriT using FPS and CLAHE")
    parser.add_argument('--img_dir', type=str, required=True, help='Directory of RGB input images (.jpg)')
    parser.add_argument('--mask_dir', type=str, required=True, help='Directory of binary masks (.png)')
    parser.add_argument('--hint_dir', type=str, required=True, help='Output directory for hint .txt files')
    parser.add_argument('--debug_dir', type=str, default=None, help='Optional: directory for visual hint overlays')
    parser.add_argument('--hint_size', type=int, default=2, help='Radius of the hint circle in pixels')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed for reproducibility')
    args = parser.parse_args()

    make_mask_guided_hints(args)
