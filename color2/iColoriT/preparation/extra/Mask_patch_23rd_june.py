import os
import cv2
import numpy as np
import pandas as pd
from skimage.color import rgb2lab

# Configuration
input_dir = "/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/data/Test/imgs_old/"
mask_dir = "/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/data/Test/masks/"
output_patch_dir = "/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/data/Test/patches"
PATCH_SIZE = 224

os.makedirs(output_patch_dir, exist_ok=True)

def generate_hint_data(rgb_img, hint_mask):
    lab_img = rgb2lab(rgb_img)
    input_l = lab_img[:, :, 0]
    ab_channels = lab_img[:, :, 1:]

    hint_ab = np.zeros_like(ab_channels)
    hint_ab[hint_mask == 1] = ab_channels[hint_mask == 1]

    hint_mask_3d = np.expand_dims(hint_mask, axis=-1)
    input_4ch = np.concatenate([input_l[..., None], hint_ab, hint_mask_3d], axis=-1)

    return input_4ch.astype(np.float32), ab_channels.astype(np.float32)

def extract_and_save_patches(input_4ch, gt_ab, base_id, patch_size=224):
    h, w, _ = input_4ch.shape
    patch_id = 1
    for i in range(0, h - patch_size + 1, patch_size):
        for j in range(0, w - patch_size + 1, patch_size):
            input_patch = input_4ch[i:i + patch_size, j:j + patch_size]
            gt_patch = gt_ab[i:i + patch_size, j:j + patch_size]

            patch_input = input_patch.transpose(2, 0, 1)
            patch_target = gt_patch.transpose(2, 0, 1)

            save_name = f"egypt_{base_id}_patch_{patch_id:02d}.npz"
            np.savez_compressed(os.path.join(output_patch_dir, save_name),
                                input=patch_input, target=patch_target)
            patch_id += 1
    return patch_id - 1

# Process all images and rename
processed_files = []
all_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".jpg")])
for idx, file_name in enumerate(all_files, start=1):
    base_name = os.path.splitext(file_name)[0]
    image_path = os.path.join(input_dir, file_name)
    mask_path = os.path.join(mask_dir, f"{base_name}.png")

    if not os.path.exists(mask_path):
        continue

    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 127).astype(np.uint8)

    input_4ch, ab_gt = generate_hint_data(image, mask)
    patch_count = extract_and_save_patches(input_4ch, ab_gt, idx, PATCH_SIZE)
    processed_files.append((f"egypt_{idx}", patch_count))
