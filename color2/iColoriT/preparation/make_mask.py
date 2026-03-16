import argparse
import os
import os.path as osp
import random
import numpy as np
from PIL import Image
from tqdm import tqdm


def make_fixed_hint(args):
    """
    Generates hints for patches of images based on colorful regions.
    """
    # Ensure output directory exists
    os.makedirs(args.hint_dir, exist_ok=True)

    # List all patch files
    patch_files = sorted([f for f in os.listdir(args.img_dir) if f.lower().endswith('.jpg')])
    random.seed(args.seed)

    # Loop through hint levels
    for num_hint in [0, 1, 2, 5, 10, 20, 50, 100, 200]:
        for patch_file in tqdm(patch_files, desc=f"Generating hints for {num_hint} hints"):
            # Load the patch
            patch_path = osp.join(args.img_dir, patch_file)
            patch = np.array(Image.open(patch_path).convert("L"))  # Load as grayscale

            # Generate binary mask for colorful regions
            binary_mask = patch > 0  # Colorful regions have pixel values > 0
            valid_coords = np.argwhere(binary_mask)  # Get (y, x) coordinates of colorful pixels

            hint_file = osp.join(args.hint_dir, f"h{args.hint_size}-n{num_hint}", f"{osp.splitext(patch_file)[0]}.txt")
            os.makedirs(osp.dirname(hint_file), exist_ok=True)

            if valid_coords.shape[0] == 0:
                # No colorful regions, create an empty hint file
                with open(hint_file, 'w') as f:
                    pass
                print(f"No colorful regions found in {patch_file}, created an empty hint file.")
                continue

            # Randomly select `num_hint` locations from the colorful regions
            selected_coords = random.sample(valid_coords.tolist(), min(num_hint, len(valid_coords)))

            # Save the hints to a text file
            lines = [f"{x} {y}\n" for y, x in selected_coords]  # Note: (row, col) -> (x, y)
            with open(hint_file, 'w') as f:
                f.writelines(lines)

    print(f"Hint generation completed. Hints saved in {args.hint_dir}")


def ensure_complete_mapping(img_dir, hint_dir):
    """
    Ensure that every image in img_dir has a corresponding hint file in hint_dir.
    If missing, create an empty hint file.
    """
    # Get image and hint filenames
    image_files = {os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.lower().endswith('.jpg')}
    hint_files = {os.path.splitext(f)[0] for f in os.listdir(hint_dir) if f.lower().endswith('.txt')}

    # Find missing hints
    missing_hints = image_files - hint_files

    for img_name in missing_hints:
        empty_hint_path = osp.join(hint_dir, f"{img_name}.txt")
        os.makedirs(osp.dirname(empty_hint_path), exist_ok=True)
        with open(empty_hint_path, 'w') as f:
            pass  # Create an empty file

    print(f"Created {len(missing_hints)} missing hint files.")


if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser(description="Making fixed hint set for interactive colorization")
    parser.add_argument('--img_dir', type=str, required=True, help="Directory containing the original patches")
    parser.add_argument('--hint_dir', type=str, required=True, help="Directory to save generated hints")
    parser.add_argument('--hint_size', type=int, default=2, help="Size of each hint region")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Generate hints
    make_fixed_hint(args)

    # Ensure all images have corresponding hints
    ensure_complete_mapping(args.img_dir, args.hint_dir)
