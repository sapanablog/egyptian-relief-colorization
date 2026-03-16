import os
import cv2
import numpy as np
from PIL import Image
import argparse
import random
from tqdm import tqdm
import json
import gc  # Garbage collection


def load_saturation_mask(mask_path):
    """
    Load saturation mask where white (255) = saturated/color part, black (0) = no color
    """
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            # Normalize to binary mask: white regions (saturated) = True, black = False
            return (mask > 127).astype(bool)
    return None


def detect_color_regions_advanced(image_rgb, threshold=20):
    """
    Advanced color detection for Egyptian artifacts (fallback method)
    """
    # Convert to different color spaces
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # Method 1: LAB color space detection
    a_channel = lab[:, :, 1].astype(np.float32)
    b_channel = lab[:, :, 2].astype(np.float32)
    color_magnitude_lab = np.sqrt((a_channel - 128) ** 2 + (b_channel - 128) ** 2)

    # Method 2: HSV saturation
    saturation = hsv[:, :, 1]

    # Method 3: RGB variance
    rgb_std = np.std(image_rgb, axis=2)

    # Combine methods
    color_mask = (
            (color_magnitude_lab > threshold) |
            (saturation > 30) |
            (rgb_std > 15)
    )

    # Morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    color_mask = cv2.morphologyEx(color_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

    return color_mask.astype(bool)


def generate_hints_from_mask(mask_patch, num_hints_list=[0, 1, 2, 5, 10, 20]):
    """
    Generate hints from saturation mask regions
    """
    # Get coordinates of saturated (colorful) pixels
    color_coords = np.argwhere(mask_patch)

    hints_dict = {}

    for num_hints in num_hints_list:
        if num_hints == 0:
            hints_dict[num_hints] = []
        elif len(color_coords) == 0:
            hints_dict[num_hints] = []
        else:
            # Sample hints from colorful regions
            if num_hints >= len(color_coords):
                selected_coords = color_coords
            else:
                # Use stratified sampling for better distribution
                indices = np.random.choice(len(color_coords), num_hints, replace=False)
                selected_coords = color_coords[indices]

            # Convert to (x, y) format
            hints_dict[num_hints] = [(coord[1], coord[0]) for coord in selected_coords]

    return hints_dict


def process_single_image_memory_efficient(image_path, mask_dir, output_dirs, split_name,
                                          patch_size=224, overlap=32, min_color_ratio=0.05):
    """
    Process a single image in a memory-efficient way
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return []

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img_rgb.shape

    print(f"Processing image: {os.path.basename(image_path)} ({h}x{w})")

    # Get corresponding mask path
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    mask_path = None
    if mask_dir:
        possible_mask_names = [
            f"{image_name}_mask.png",
            f"{image_name}_mask.jpg",
            f"{image_name}.png",
            f"{image_name}.jpg",
            f"{image_name}_saturation.png",
            f"{image_name}_saturation.jpg"
        ]

        for mask_name in possible_mask_names:
            potential_path = os.path.join(mask_dir, mask_name)
            if os.path.exists(potential_path):
                mask_path = potential_path
                break

    # Load full mask once
    full_mask = None
    if mask_path:
        full_mask = load_saturation_mask(mask_path)
        if full_mask is not None and full_mask.shape != (h, w):
            full_mask = cv2.resize(
                full_mask.astype(np.uint8), (w, h),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

    # Calculate stride
    stride = patch_size - overlap

    patches_info = []
    patch_count = 0

    # Process patches one by one to save memory
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            # Extract patch
            patch = img_rgb[y:y + patch_size, x:x + patch_size]

            # Extract mask patch
            if full_mask is not None:
                mask_patch = full_mask[y:y + patch_size, x:x + patch_size]
            else:
                mask_patch = detect_color_regions_advanced(patch)

            # Check if patch has enough color content
            color_ratio = np.sum(mask_patch) / (patch_size * patch_size)

            if color_ratio >= min_color_ratio:
                patch_name = f"{image_name}_patch_{patch_count:04d}"

                # Save patch immediately
                patch_path = os.path.join(output_dirs['patches'], f"{patch_name}.jpg")
                Image.fromarray(patch).save(patch_path, quality=95)

                # Save mask patch for debugging
                mask_path_save = os.path.join(output_dirs['masks'], f"{patch_name}_mask.png")
                Image.fromarray((mask_patch * 255).astype(np.uint8)).save(mask_path_save)

                # Generate and save hints
                hints_dict = generate_hints_from_mask(mask_patch)
                for level, coords in hints_dict.items():
                    hint_file = os.path.join(output_dirs['hints'][level], f"{patch_name}.txt")
                    with open(hint_file, 'w') as f:
                        for hx, hy in coords:
                            f.write(f"{hx} {hy}\n")

                # Store patch info
                patches_info.append({
                    'patch_name': patch_name,
                    'original_image': os.path.basename(image_path),
                    'position': (x, y, os.path.basename(image_path)),
                    'color_ratio': color_ratio
                })

                patch_count += 1

                # Clear variables to free memory
                del patch, mask_patch

    # Clear large variables
    del img_rgb, full_mask
    gc.collect()  # Force garbage collection

    print(f"  Extracted {len(patches_info)} patches from {os.path.basename(image_path)}")
    return patches_info


def process_dataset_split(input_dir, mask_dir, output_dir, split_name,
                          patch_size=224, overlap=32, min_color_ratio=0.05):
    """
    Process a dataset split in a memory-efficient way
    """
    print(f"\nProcessing {split_name} dataset...")

    # Create directories
    patches_dir = os.path.join(output_dir, "patches", split_name, "images")
    masks_dir = os.path.join(output_dir, "masks", split_name)
    positions_dir = os.path.join(output_dir, "positions", split_name)

    os.makedirs(patches_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(positions_dir, exist_ok=True)

    # Create hint directories
    hint_levels = [0, 1, 2, 5, 10, 20]
    hint_dirs = {}
    for level in hint_levels:
        hint_dir = os.path.join(output_dir, "hints", split_name, f"h2-n{level}")
        os.makedirs(hint_dir, exist_ok=True)
        hint_dirs[level] = hint_dir

    # Prepare output directories dict
    output_dirs = {
        'patches': patches_dir,
        'masks': masks_dir,
        'hints': hint_dirs
    }

    # Get all images
    image_files = [f for f in os.listdir(input_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif'))]

    print(f"Found {len(image_files)} images in {input_dir}")
    if mask_dir and os.path.exists(mask_dir):
        print(f"Using masks from: {mask_dir}")
    else:
        print("No mask directory found, using fallback color detection")

    all_patches_info = []

    # Process images one by one
    for img_file in tqdm(image_files, desc=f"Processing {split_name}"):
        img_path = os.path.join(input_dir, img_file)

        # Process single image
        patches_info = process_single_image_memory_efficient(
            img_path, mask_dir, output_dirs, split_name,
            patch_size, overlap, min_color_ratio
        )

        all_patches_info.extend(patches_info)

        # Force garbage collection after each image
        gc.collect()

    # Save patch information
    with open(os.path.join(positions_dir, f"{split_name}_patches_info.json"), 'w') as f:
        json.dump(all_patches_info, f, indent=2)

    # Generate statistics
    if all_patches_info:
        avg_color_ratio = np.mean([p['color_ratio'] for p in all_patches_info])
        print(f"\n{split_name.upper()} dataset statistics:")
        print(f"  Total patches: {len(all_patches_info)}")
        print(f"  Average color ratio: {avg_color_ratio:.4f}")

        # Show hint statistics
        print(f"  Hint statistics:")
        for level in hint_levels:
            hint_dir = hint_dirs[level]
            hint_files = [f for f in os.listdir(hint_dir) if f.endswith('.txt')]
            total_hints = 0
            non_empty_files = 0

            for hint_file in hint_files:
                with open(os.path.join(hint_dir, hint_file), 'r') as f:
                    lines = f.readlines()
                    if lines:
                        total_hints += len(lines)
                        non_empty_files += 1

            avg_hints = total_hints / max(non_empty_files, 1)
            print(
                f"    Level {level}: {non_empty_files}/{len(hint_files)} patches with hints, avg {avg_hints:.2f} hints/patch")

    return all_patches_info


def validate_dataset_split(output_dir, split_name):
    """
    Validate a single dataset split
    """
    print(f"\nValidating {split_name} dataset...")

    required_dirs = [
        f"patches/{split_name}/images",
        f"hints/{split_name}/h2-n0",
        f"hints/{split_name}/h2-n1",
        f"hints/{split_name}/h2-n2",
        f"hints/{split_name}/h2-n5",
        f"hints/{split_name}/h2-n10",
        f"hints/{split_name}/h2-n20",
        f"positions/{split_name}"
    ]

    for dir_path in required_dirs:
        full_path = os.path.join(output_dir, dir_path)
        if not os.path.exists(full_path):
            print(f"Error: Missing directory {full_path}")
            return False
        else:
            file_count = len([f for f in os.listdir(full_path) if not f.startswith('.')])
            print(f"✓ {dir_path}: {file_count} files")

    print(f"{split_name} dataset validation completed!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory-efficient Egyptian dataset preparation")
    parser.add_argument('--train_dir', type=str, help='Directory containing training images')
    parser.add_argument('--val_dir', type=str, help='Directory containing validation images')
    parser.add_argument('--test_dir', type=str, help='Directory containing test images')
    parser.add_argument('--train_mask_dir', type=str, help='Directory containing training masks')
    parser.add_argument('--val_mask_dir', type=str, help='Directory containing validation masks')
    parser.add_argument('--test_mask_dir', type=str, help='Directory containing test masks')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for prepared dataset')
    parser.add_argument('--patch_size', type=int, default=224,
                        help='Size of patches to extract')
    parser.add_argument('--overlap', type=int, default=32,
                        help='Overlap between patches')
    parser.add_argument('--min_color_ratio', type=float, default=0.05,
                        help='Minimum ratio of colored pixels in patch to include it')
    parser.add_argument('--validate', action='store_true',
                        help='Validate dataset after preparation')

    args = parser.parse_args()

    # Create main output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each dataset split separately
    if args.train_dir and os.path.exists(args.train_dir):
        print("Starting training dataset processing...")
        train_info = process_dataset_split(
            args.train_dir, args.train_mask_dir, args.output_dir,
            'train', args.patch_size, args.overlap, args.min_color_ratio
        )

        if args.validate:
            validate_dataset_split(args.output_dir, 'train')

    if args.val_dir and os.path.exists(args.val_dir):
        print("Starting validation dataset processing...")
        val_info = process_dataset_split(
            args.val_dir, args.val_mask_dir, args.output_dir,
            'val', args.patch_size, args.overlap, args.min_color_ratio
        )

        if args.validate:
            validate_dataset_split(args.output_dir, 'val')

    if args.test_dir and os.path.exists(args.test_dir):
        print("Starting test dataset processing...")
        test_info = process_dataset_split(
            args.test_dir, args.test_mask_dir, args.output_dir,
            'test', args.patch_size, args.overlap, args.min_color_ratio
        )

        if args.validate:
            validate_dataset_split(args.output_dir, 'test')

    print(f"\nAll dataset preparation completed!")
    print(f"Output directory: {args.output_dir}")