import argparse
import os
import os.path as osp
import random
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import matplotlib.pyplot as plt


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
            #mask_path = osp.join(args.mask_dir, filename)
            mask_filename = osp.splitext(filename)[0] + ".png"
            mask_path = osp.join(args.mask_dir, mask_filename)

            # Load RGB image and binary mask
            image = Image.open(img_path).convert('RGB').resize((224, 224), Image.BICUBIC)
            mask = Image.open(mask_path).convert('L').resize((224, 224), Image.NEAREST)
            # mask = Image.open(mask_path).convert('L')

            # Resize both image and mask to 224x224
            image = np.array(image)
            mask = np.array(mask)

            image_np = image
            mask_np = mask

            # Find valid hint locations (where mask > 0)
            valid_coords = np.argwhere(mask_np > 0)  # (y, x)

            renamed_base = f"egypt_{idx+1}"
            hint_txt_name = f"{renamed_base}.txt"
            hint_txt_path = osp.join(hint_subdir, hint_txt_name)

            selected_coords = []
            if valid_coords.shape[0] > 0 and num_hint > 0:
                selected_coords = random.sample(valid_coords.tolist(), min(num_hint, len(valid_coords)))
                with open(hint_txt_path, 'w') as f:
                    for y, x in selected_coords:
                        f.write(f"{x} {y}\n")
            else:
                with open(hint_txt_path, 'w') as f:
                    pass  # empty hint file

            # Optional: save debug image
            if args.debug_dir:
                debug_img = Image.fromarray(image_np.copy())
                draw = ImageDraw.Draw(debug_img)
                for y, x in selected_coords:
                    r = args.hint_size
                    draw.ellipse([(x - r, y - r), (x + r, y + r)], outline='red', width=1)
                debug_name = f"{renamed_base}.jpg"
                debug_img.save(osp.join(args.debug_dir, f"{num_hint}_hints", debug_name))

            # Visualize side-by-side for first image only and 20 hints
            if idx == 0 and num_hint == 20:
                fig, axs = plt.subplots(2, 2, figsize=(12, 12))

                # Original image
                axs[0, 0].imshow(image_np)
                axs[0, 0].set_title("Original Image")
                axs[0, 0].axis('off')

                # Binary mask
                axs[0, 1].imshow(mask_np, cmap='gray')
                axs[0, 1].set_title("Binary Mask")
                axs[0, 1].axis('off')

                # Hint mask with red dots
                hint_mask = np.zeros_like(image_np)
                for y, x in selected_coords:
                    r = args.hint_size
                    hint_mask[y - r:y + r + 1, x - r:x + r + 1] = [255, 0, 0]  # red square
                axs[1, 0].imshow(hint_mask)
                axs[1, 0].set_title("Hints Only (Red Blocks)")
                axs[1, 0].axis('off')

                # Overlay blocks on image
                overlay_np = image_np.copy()
                for y, x in selected_coords:
                    r = args.hint_size
                    overlay_np[y - r:y + r + 1, x - r:x + r + 1] = [0, 255, 0]  # green square
                axs[1, 1].imshow(overlay_np)
                axs[1, 1].set_title("Hints Overlay (Green Blocks)")
                axs[1, 1].axis('off')

                plt.tight_layout()
                vis_path = osp.join(args.debug_dir, f"{num_hint}_hints", f"{renamed_base}_vis.jpg")
                plt.savefig(vis_path)
                plt.close()

    print("Hint generation complete.")


def ensure_complete_mapping(img_dir, hint_dir):
    image_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')])
    expected_hint_files = {f"egypt_{idx+1}" for idx in range(len(image_files))}
    actual_hint_files = {osp.splitext(f)[0] for f in os.listdir(hint_dir) if f.lower().endswith('.txt')}

    missing = expected_hint_files - actual_hint_files
    for name in missing:
        path = osp.join(hint_dir, f"{name}.txt")
        os.makedirs(osp.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            pass
    print(f"Created {len(missing)} missing hint files.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate hint maps from RGB images and binary masks")
    parser.add_argument('--img_dir', type=str, required=True, help='Directory of RGB images')
    parser.add_argument('--mask_dir', type=str, required=True, help='Directory of binary mask images')
    parser.add_argument('--hint_dir', type=str, required=True, help='Output directory for hint txts')
    parser.add_argument('--debug_dir', type=str, default=None, help='Directory to save visual debug overlays')
    parser.add_argument('--hint_size', type=int, default=2, help='Size of each hint point (visual)')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    args = parser.parse_args()

    make_mask_guided_hints(args)

    for num_hint in [0, 1, 2, 5, 10, 20]:
        sub_hint_dir = osp.join(args.hint_dir, f"h{args.hint_size}-n{num_hint}")
        ensure_complete_mapping(args.img_dir, sub_hint_dir)
