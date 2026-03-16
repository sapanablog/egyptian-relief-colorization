import argparse
import os
import os.path as osp
from PIL import Image
import numpy as np
from tqdm import tqdm
import random


def snap_to_grid(coords_xy, step):
    """
    Snap integer (x, y) coordinates to the nearest lower multiple of `step`.
    Keeps coordinates within image bounds later by clipping.
    """
    coords_xy = np.asarray(coords_xy, dtype=np.int32)
    coords_xy //= step
    coords_xy *= step
    return coords_xy


def top_saturation_points(img_rgb, mask_bin, top_k=20, min_value=0, min_saturation=0, hint_size=2, seed=1234):
    """
    Select up to top_k points with highest saturation inside mask_bin.
    - img_rgb: HxWx3 uint8
    - mask_bin: HxW boolean
    - min_value: ignore pixels with V < min_value
    - min_saturation: ignore pixels with S < min_saturation (0-255)
    - hint_size: snap to this grid (e.g., 2)
    Returns: list of (x, y) integer tuples, length <= top_k
    """
    # Convert to HSV
    pil = Image.fromarray(img_rgb, mode="RGB").convert("HSV")
    hsv = np.array(pil)  # H,S,V in 0..255
    S = hsv[..., 1].astype(np.int16)
    V = hsv[..., 2].astype(np.int16)

    # Valid region: mask & brightness & saturation thresholds
    valid = mask_bin & (V >= int(min_value)) & (S >= int(min_saturation))
    if not np.any(valid):
        return []

    # Get candidate coords (y, x) and their S
    ys, xs = np.where(valid)
    sats = S[ys, xs]

    # Sort by saturation desc (stable, deterministic)
    # To add a tiny deterministic shuffle on ties, we can mix in indices with a fixed seed.
    rng = np.random.default_rng(seed)
    tie_breaker = rng.random(len(sats)) * 1e-6
    order = np.argsort(-(sats + tie_breaker))

    xs_sorted = xs[order]
    ys_sorted = ys[order]

    # Snap to grid and deduplicate after snapping
    snapped = snap_to_grid(np.stack([xs_sorted, ys_sorted], axis=1), hint_size)
    # Deduplicate while preserving order
    seen = set()
    unique_xy = []
    for x, y in snapped:
        key = (int(x), int(y))
        if key not in seen:
            seen.add(key)
            unique_xy.append(key)
        if len(unique_xy) >= top_k:
            break

    return unique_xy


def write_hierarchical_hints(points_xy, out_root, base_stem, hint_size=2):
    """
    Write cumulative subsets of points to:
      out_root/h{hint_size}-n{N}/{base_stem}.txt
    For N in [0,1,2,5,10,20].
    Each line: "x y"
    """
    levels = [0, 1, 2, 5, 10, 20]
    for N in levels:
        subdir = osp.join(out_root, f"h{hint_size}-n{N}")
        os.makedirs(subdir, exist_ok=True)
        out_path = osp.join(subdir, f"{base_stem}.txt")
        with open(out_path, "w") as f:
            for (x, y) in points_xy[:N]:
                f.write(f"{x} {y}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate saturated-region hints from masked Egyptian relief images.")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory with input images (e.g., Train)")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory with binary masks (e.g., Train_masks)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output root directory for hints")
    parser.add_argument("--img_ext", type=str, default=".jpg", help="Image extension (e.g., .jpg)")
    parser.add_argument("--mask_ext", type=str, default=".png", help="Mask extension (e.g., .png)")
    parser.add_argument("--hint_size", type=int, default=2, help="Grid snapping step (default 2)")
    parser.add_argument("--top_k", type=int, default=20, help="Maximum number of hints to extract (default 20)")
    parser.add_argument("--min_value", type=int, default=30, help="Min V (brightness) in HSV to consider (0-255)")
    parser.add_argument("--min_saturation", type=int, default=10, help="Min S (saturation) in HSV to consider (0-255)")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for tie-breaking")
    args = parser.parse_args()

    # List images, only those that have a matching mask by stem
    img_files = sorted([f for f in os.listdir(args.img_dir) if f.lower().endswith(args.img_ext)])
    if not img_files:
        raise RuntimeError(f"No images with ext {args.img_ext} found in {args.img_dir}")

    paired = []
    for f in img_files:
        stem = osp.splitext(f)[0]  # egypt_01
        mask_path = osp.join(args.mask_dir, stem + args.mask_ext)
        if osp.exists(mask_path):
            paired.append((osp.join(args.img_dir, f), mask_path, stem))
        else:
            # Skip silently, or print a warning:
            print(f" No matching mask for {f} in {args.mask_dir}; skipping.")

    if not paired:
        raise RuntimeError("No image/mask pairs found. Check filenames and extensions.")

    random.seed(args.seed)
    np.random.seed(args.seed)

    for img_path, mask_path, stem in tqdm(paired, desc="Processing"):
        # Load image and mask
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img_np = np.array(img)  # HxWx3 uint8
        mask_np = np.array(mask)  # HxW uint8
        mask_bin = mask_np > 0

        # Pick top saturated points within the mask
        points = top_saturation_points(
            img_np,
            mask_bin,
            top_k=args.top_k,
            min_value=args.min_value,
            min_saturation=args.min_saturation,
            hint_size=args.hint_size,
            seed=args.seed,
        )

        # Write cumulative files
        write_hierarchical_hints(points, args.out_dir, stem, hint_size=args.hint_size)

    print(" Done. Hints written to:", args.out_dir)


if __name__ == "__main__":
    main()
