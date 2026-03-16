# # ###this is for rgb
# #
# # import os
# # import numpy as np
# # from PIL import Image
# # from skimage.color import rgb2lab, lab2rgb
# # import torch
# # import sys
# # import modeling
# # from timm.models import create_model
# #
# # # ==== SETUP ====
# # sys.path.append('/home/sapanagupta/PycharmProjects/color2/iColoriT/')
# # PATCH_SIZE = 224
# # DEVICE = 'cuda'
# # MODEL_NAME = 'icolorit_base_4ch_patch16_224'
# # HEAD_MODE = 'cnn'
# #
# # # ==== PATHS ====
# # CKPT_PATH = '/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/demo_latest/Train/output_dir/icolorit_base_4ch_patch16_224/patchwise_egyptian_250614_161118/checkpoint-49.pth'
# # ORIG_IMG_DIR = '/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/demo_latest/Test/imgs_old/'
# # IMG_PATCH_DIR = '/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/demo_latest/Test/imgs/'
# # AB_HINT_PATCH_DIR = '/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/demo_latest/Test/hint_ab_patches/'
# # MASK_PATCH_DIR = '/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/demo_latest/Test/hint_masks_patches/'
# # SAVE_DIR = '/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/demo_latest/Test/prediction'
# #
# # os.makedirs(SAVE_DIR, exist_ok=True)
# #
# # # ==== LOAD MODEL ====
# # model = create_model(MODEL_NAME, pretrained=False, head_mode=HEAD_MODE)
# # checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
# # model.load_state_dict(checkpoint['model'], strict=False)
# # model.to(DEVICE)
# # model.eval()
# #
# # # ==== HELPERS ====
# # def load_npz_arr(npz_path):
# #     data = np.load(npz_path)
# #     if 'arr' not in data:
# #         raise KeyError(f"Key 'arr' not found in {npz_path}. Available keys: {list(data.keys())}")
# #     return data['arr']
# #
# # def infer_patch(model, l_patch, ab_hint, mask):
# #     input_tensor = np.zeros((4, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
# #     input_tensor[0] = l_patch
# #     input_tensor[1:3] = ab_hint
# #     input_tensor[3] = mask
# #     input_tensor = torch.tensor(input_tensor).unsqueeze(0).to(DEVICE)
# #     patch_mask = torch.ones((1, 1), dtype=torch.float32, device=input_tensor.device)
# #
# #     with torch.no_grad():
# #         output = model(input_tensor, patch_mask)
# #         ab_pred = output.reshape(PATCH_SIZE, PATCH_SIZE, 2).cpu().numpy()
# #     return ab_pred
# #
# # def reconstruct_full_image(image_id):
# #     orig_img_path = os.path.join(ORIG_IMG_DIR, f'{image_id}.jpg')
# #     if not os.path.exists(orig_img_path):
# #         print(f"[SKIP] Missing: {orig_img_path}")
# #         return
# #
# #     orig_img = np.array(Image.open(orig_img_path).convert('RGB')) / 255.0
# #     lab_img = rgb2lab(orig_img).astype(np.float32)
# #     L_full = lab_img[..., 0] / 100.0
# #
# #     H_full, W_full = L_full.shape
# #     ab_full = np.zeros((H_full, W_full, 2), dtype=np.float32)
# #
# #     n_patches_x = int(np.ceil(W_full / PATCH_SIZE))
# #     n_patches_y = int(np.ceil(H_full / PATCH_SIZE))
# #     idx = 1
# #
# #     for py in range(n_patches_y):
# #         for px in range(n_patches_x):
# #             patch_name = f"{image_id}_patch_{idx:02d}"
# #             img_patch_path = os.path.join(IMG_PATCH_DIR, patch_name + '.jpg')
# #             ab_npz_path = os.path.join(AB_HINT_PATCH_DIR, patch_name + '.npz')
# #             mask_npz_path = os.path.join(MASK_PATCH_DIR, patch_name + '.npz')
# #
# #             if not (os.path.exists(img_patch_path) and os.path.exists(ab_npz_path) and os.path.exists(mask_npz_path)):
# #                 idx += 1
# #                 continue
# #
# #             img_patch = np.array(Image.open(img_patch_path).convert('RGB')) / 255.0
# #             l_patch = rgb2lab(img_patch).astype(np.float32)[..., 0] / 100.0
# #
# #             ab_hint = load_npz_arr(ab_npz_path)
# #             mask = load_npz_arr(mask_npz_path)
# #             if mask.ndim == 3:
# #                 mask = mask[0]
# #
# #             ab_pred = infer_patch(model, l_patch, ab_hint, mask)
# #
# #             y0, x0 = py * PATCH_SIZE, px * PATCH_SIZE
# #             h = min(PATCH_SIZE, H_full - y0)
# #             w = min(PATCH_SIZE, W_full - x0)
# #             ab_full[y0:y0+h, x0:x0+w, :] = ab_pred[:h, :w, :]
# #             idx += 1
# #
# #     lab = np.zeros((H_full, W_full, 3), dtype=np.float32)
# #     lab[..., 0] = L_full * 100.0
# #     lab[..., 1:] = ab_full * 128.0  # 🔥 SCALE BACK TO LAB RANGE
# #
# #     rgb = (lab2rgb(lab) * 255).clip(0, 255).astype(np.uint8)
# #     out_path = os.path.join(SAVE_DIR, f'{image_id}_colorized.png')
# #     Image.fromarray(rgb).save(out_path)
# #     print(f"✅ Saved: {out_path}")
# #
# # # ==== RUN ====
# # image_ids = [f.split('.')[0] for f in os.listdir(ORIG_IMG_DIR) if f.endswith('.jpg')]
# # image_ids.sort()
# # for image_id in image_ids:
# #     reconstruct_full_image(image_id)
#
#
#
# # ############this is for lab becoz icolorit  takes lab
# # import os
# # import numpy as np
# # from PIL import Image
# # from skimage.color import rgb2lab, lab2rgb
# # import torch
# # import sys
# # from timm.models import create_model
# # import modeling  # Required to register IColoriT
# #
# # # ==== SETUP ====
# # sys.path.append('/home/sapanagupta/PycharmProjects/color2/iColoriT/')
# # PATCH_SIZE = 224
# # DEVICE = 'cuda'
# # MODEL_NAME = 'icolorit_base_4ch_patch16_224'
# # HEAD_MODE = 'cnn'
# #
# # # ==== PATHS ====
# # CKPT_PATH = '/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/demo_latest_new/Train/output_dir/icolorit_base_4ch_patch16_224/patchwise_egyptian_250615_135505/checkpoint-49.pth'
# # ORIG_IMG_DIR = '/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/demo_latest_new/Test/imgs_old/'
# # IMG_PATCH_DIR = '/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/demo_latest_new/Test/imgs/'
# # AB_HINT_PATCH_DIR = '/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/demo_latest_new/Test/hint_ab_patches/'
# # MASK_PATCH_DIR = '/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/demo_latest_new/Test/hint_masks_patches/'
# # SAVE_DIR = '/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/demo_latest_new/Test/prediction'
# #
# # os.makedirs(SAVE_DIR, exist_ok=True)
# #
# # # ==== LOAD MODEL ====
# # model = create_model(MODEL_NAME, pretrained=False, head_mode=HEAD_MODE)
# # checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
# # model.load_state_dict(checkpoint['model'], strict=False)
# # model.to(DEVICE)
# # model.eval()
# #
# # # ==== HELPERS ====
# # def load_npz_arr(npz_path):
# #     data = np.load(npz_path)
# #     if 'arr' not in data:
# #         raise KeyError(f"Key 'arr' not found in {npz_path}. Keys: {list(data.keys())}")
# #     return data['arr']
# #
# # def infer_patch(model, l_patch, ab_hint, mask):
# #     input_tensor = np.zeros((4, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
# #     input_tensor[0] = l_patch
# #     input_tensor[1:3] = ab_hint
# #     input_tensor[3] = mask
# #     input_tensor = torch.tensor(input_tensor).unsqueeze(0).to(DEVICE)
# #     patch_mask = torch.ones((1, 1), dtype=torch.float32, device=input_tensor.device)
# #
# #     with torch.no_grad():
# #         output = model(input_tensor, patch_mask)
# #         ab_pred = output.reshape(PATCH_SIZE, PATCH_SIZE, 2).cpu().numpy()
# #     return ab_pred
# #
# # def reconstruct_full_image(image_id):
# #     orig_img_path = os.path.join(ORIG_IMG_DIR, f'{image_id}.jpg')
# #     if not os.path.exists(orig_img_path):
# #         print(f"[SKIP] Image not found: {orig_img_path}")
# #         return
# #
# #     orig_img = Image.open(orig_img_path).convert('RGB')
# #     W_full, H_full = orig_img.size
# #     lab_img = rgb2lab(np.array(orig_img) / 255.0).astype(np.float32)
# #     L_full = lab_img[..., 0]
# #     ab_full = np.zeros((H_full, W_full, 2), dtype=np.float32)
# #
# #     n_patches_x = int(np.ceil(W_full / PATCH_SIZE))
# #     n_patches_y = int(np.ceil(H_full / PATCH_SIZE))
# #     idx = 1
# #
# #     for py in range(n_patches_y):
# #         for px in range(n_patches_x):
# #             patch_name = f"{image_id}_patch_{idx:02d}"
# #             img_patch_path = os.path.join(IMG_PATCH_DIR, patch_name + '.jpg')
# #             ab_npz_path = os.path.join(AB_HINT_PATCH_DIR, patch_name + '.npz')
# #             mask_npz_path = os.path.join(MASK_PATCH_DIR, patch_name + '.npz')
# #
# #             if not all(map(os.path.exists, [img_patch_path, ab_npz_path, mask_npz_path])):
# #                 idx += 1
# #                 continue
# #
# #             img_patch = np.array(Image.open(img_patch_path).convert('RGB')) / 255.0
# #             l_patch = rgb2lab(img_patch).astype(np.float32)[..., 0]
# #             ab_hint = load_npz_arr(ab_npz_path)
# #             mask = load_npz_arr(mask_npz_path)
# #             if mask.ndim == 3:
# #                 mask = mask[0]
# #
# #             ab_pred = infer_patch(model, l_patch, ab_hint, mask)
# #             y0, x0 = py * PATCH_SIZE, px * PATCH_SIZE
# #             h, w = min(PATCH_SIZE, H_full - y0), min(PATCH_SIZE, W_full - x0)
# #             ab_full[y0:y0+h, x0:x0+w, :] = ab_pred[:h, :w, :]
# #             idx += 1
# #
# #     lab = np.zeros((H_full, W_full, 3), dtype=np.float32)
# #     lab[..., 0] = L_full
# #     lab[..., 1:] = ab_full
# #     rgb = (lab2rgb(lab) * 255).astype(np.uint8)
# #
# #     out_path = os.path.join(SAVE_DIR, f'{image_id}_colorized.png')
# #     Image.fromarray(rgb).save(out_path)
# #     print(f"✅ Saved: {out_path}")
# #
# # # ==== RUN ====
# # image_ids = [f.split('.')[0] for f in os.listdir(ORIG_IMG_DIR) if f.endswith('.jpg')]
# # image_ids.sort()
# # for image_id in image_ids:
# #     reconstruct_full_image(image_id)
# #
# #
# #
#
#
# #######this is using 40 % portion
#
# import os
# import numpy as np
# from PIL import Image
# from skimage.color import rgb2lab
# from skimage.util import view_as_blocks
# import pandas as pd
#
# # ==== CONFIGURATION ====
# PATCH_SIZE = 224
# HINT_PORTION = 0.95 # 40% of colored region used as hint
#
# # ==== INPUT DIRECTORY ====
# source_img_dir = "/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/demo_latest_new/Valid/imgs_old"
#
# # ==== OUTPUT DIRECTORIES ====
# save_img_patch_dir = "/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/demo_latest_new/Valid/imgs/"
# save_hint_ab_dir = "/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/demo_latest_new/Valid/hint_ab_patches/"
# save_hint_mask_dir = "/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/demo_latest_new/Valid/hint_masks_patches/"
#
# os.makedirs(save_img_patch_dir, exist_ok=True)
# os.makedirs(save_hint_ab_dir, exist_ok=True)
# os.makedirs(save_hint_mask_dir, exist_ok=True)
#
# # ==== IMAGE LIST ====
# source_img_paths = sorted([
#     os.path.join(source_img_dir, f)
#     for f in os.listdir(source_img_dir)
#     if f.lower().endswith((".jpg", ".jpeg", ".png"))
# ])
#
# # ==== PATCHIFY UTILS ====
# def patchify(img, patch_size):
#     h, w = img.shape[:2]
#     h_crop = h - h % patch_size
#     w_crop = w - w % patch_size
#     img_cropped = img[:h_crop, :w_crop]
#     return view_as_blocks(img_cropped, block_shape=(patch_size, patch_size, img.shape[2])).reshape(-1, patch_size, patch_size, img.shape[2])
#
# def patchify_single_channel(img, patch_size):
#     h, w = img.shape[1:]
#     h_crop = h - h % patch_size
#     w_crop = w - w % patch_size
#     img_cropped = img[:, :h_crop, :w_crop]
#     return view_as_blocks(img_cropped[0], block_shape=(patch_size, patch_size)).reshape(-1, patch_size, patch_size)
#
# def patchify_ab(ab, patch_size):
#     h, w = ab.shape[1:]
#     h_crop = h - h % patch_size
#     w_crop = w - w % patch_size
#     ab_cropped = ab[:, :h_crop, :w_crop]
#     ab_patches = []
#     for i in range(2):
#         ab_patches.append(view_as_blocks(ab_cropped[i], block_shape=(patch_size, patch_size)).reshape(-1, patch_size, patch_size))
#     return np.stack(ab_patches, axis=1)
#
# # ==== PROCESSING LOOP ====
# meta = {"Patch": [], "Hint_ab": [], "Hint_mask": []}
#
# for img_idx, img_path in enumerate(source_img_paths):
#     base_name = f"{img_idx+1:03d}"  # egypt_001, egypt_002 ...
#     img = Image.open(img_path).convert('RGB')
#     img_np = np.array(img).astype(np.float32) / 255.0
#
#     # Convert to Lab and detect colored regions
#     lab = rgb2lab(img_np)
#     L, A, B = lab[..., 0], lab[..., 1] / 128.0, lab[..., 2] / 128.0
#
#     r, g, b = img_np[..., 0], img_np[..., 1], img_np[..., 2]
#     mask_color = ~((np.abs(r - g) < 1e-4) & (np.abs(g - b) < 1e-4))
#
#     # Generate hint mask by randomly sampling colored pixels
#     ab_hint = np.zeros((2, *mask_color.shape), dtype=np.float32)
#     hint_mask = np.zeros((1, *mask_color.shape), dtype=np.float32)
#
#     coords = np.argwhere(mask_color)
#     np.random.shuffle(coords)
#     keep_coords = coords[:int(len(coords) * HINT_PORTION)]
#
#     for y, x in keep_coords:
#         ab_hint[0, y, x] = A[y, x]
#         ab_hint[1, y, x] = B[y, x]
#         hint_mask[0, y, x] = 1.0
#
#     # Patchify all
#     img_patches = patchify(img_np, PATCH_SIZE)
#     ab_patches = patchify_ab(ab_hint, PATCH_SIZE)
#     mask_patches = patchify_single_channel(hint_mask, PATCH_SIZE)
#
#     for i, (img_patch, ab_patch, mask_patch) in enumerate(zip(img_patches, ab_patches, mask_patches)):
#         patch_index = i + 1
#         patch_name = f"egypt_{base_name}_patch_{patch_index:02d}"
#
#         img_save_path = os.path.join(save_img_patch_dir, patch_name + ".jpg")
#         ab_save_path = os.path.join(save_hint_ab_dir, patch_name + ".npz")
#         mask_save_path = os.path.join(save_hint_mask_dir, patch_name + ".npz")
#
#         Image.fromarray((img_patch * 255).astype(np.uint8)).save(img_save_path)
#         np.savez_compressed(ab_save_path, arr=ab_patch)
#         np.savez_compressed(mask_save_path, arr=mask_patch[np.newaxis])
#
#         meta["Patch"].append(patch_name + ".jpg")
#         meta["Hint_ab"].append(patch_name + ".npz")
#         meta["Hint_mask"].append(patch_name + ".npz")
#
# # Save metadata CSV
# df = pd.DataFrame(meta)
# df.to_csv("/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/demo_latest_new/patch_metadata.csv", index=False)
# print("✅ All patch-wise images, ab hints, and masks saved with partial color hints.")
#


'''
#######################
import os
import numpy as np
from PIL import Image
from skimage.color import lab2rgb, rgb2lab
import torch
from timm.models import create_model
import modeling  # Required to register IColoriT

# ==== CONFIGURATION ====
PATCH_SIZE = 224
DEVICE = 'cuda'
MODEL_NAME = 'icolorit_base_4ch_patch16_224'
HEAD_MODE = 'cnn'

# ==== PATHS ====
CKPT_PATH = '/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/data_icolorit/Train/output_dir/icolorit_base_4ch_patch16_224/patchwise_egyptian_250616_155029/checkpoint-49.pth'
IMG_PATCH_DIR = '/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/data_icolorit/Test/imgs/'
AB_HINT_PATCH_DIR = '/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/data_icolorit/Test/hint_ab_patches/'
MASK_PATCH_DIR = '/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/data_icolorit/Test/hint_masks_patches/'
SAVE_DIR = '/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/data_icolorit/Test/prediction/'

os.makedirs(SAVE_DIR, exist_ok=True)

# ==== LOAD MODEL ====
model = create_model(MODEL_NAME, pretrained=False, head_mode=HEAD_MODE)
checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model'], strict=False)
model.to(DEVICE)
model.eval()


# ==== HELPERS ====
def load_npz_arr(npz_path):
    data = np.load(npz_path)
    if 'arr' not in data:
        raise KeyError(f"Missing key 'arr' in: {npz_path}")
    return data['arr']


def infer_patch(model, l_patch, ab_hint, mask):
    input_tensor = np.zeros((4, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    input_tensor[0] = l_patch
    input_tensor[1:3] = ab_hint
    input_tensor[3] = mask
    input_tensor = torch.tensor(input_tensor).unsqueeze(0).to(DEVICE)
    patch_mask = torch.ones((1, 1), dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        output = model(input_tensor, patch_mask)
        ab_pred = output.reshape(PATCH_SIZE, PATCH_SIZE, 2).cpu().numpy()
    return ab_pred


# ==== INFERENCE LOOP ====
patch_names = sorted([
    f for f in os.listdir(IMG_PATCH_DIR)
    if f.endswith(".jpg") and f.startswith("egypt_")
])

for patch_name in patch_names:
    base_name = patch_name.replace(".jpg", "")
    img_path = os.path.join(IMG_PATCH_DIR, patch_name)
    ab_hint_path = os.path.join(AB_HINT_PATCH_DIR, base_name + ".npz")
    mask_path = os.path.join(MASK_PATCH_DIR, base_name + ".npz")

    if not all(map(os.path.exists, [img_path, ab_hint_path, mask_path])):
        print(f"[Skip] Missing one or more inputs for {base_name}")
        continue

    # Load inputs
    img_patch = np.array(Image.open(img_path).convert('RGB')) / 255.0
    l_patch = rgb2lab(img_patch)[..., 0] / 100.0
    ab_hint = load_npz_arr(ab_hint_path)
    hint_mask = load_npz_arr(mask_path)
    if hint_mask.ndim == 3:
        hint_mask = hint_mask[0]

    # Predict ab channels
    ab_pred = infer_patch(model, l_patch, ab_hint, hint_mask)

    # Convert to RGB and save
    lab_patch = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.float32)
    lab_patch[..., 0] = l_patch * 100.0
    lab_patch[..., 1:] = ab_pred * 128.0
    rgb_patch = (lab2rgb(lab_patch) * 255).astype(np.uint8)

    save_path = os.path.join(SAVE_DIR, base_name + "_colorized_patch.png")
    Image.fromarray(rgb_patch).save(save_path)
    print(f"✅ Saved: {save_path}")


'''


###this is for full resolution for mask genertated from resnet50
# File: infer.py
import argparse
import os
import os.path as osp
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from timm.models import create_model

# --- THIS IS THE FIX ---
from einops import rearrange
# --- END OF FIX ---

import modeling  # To register models
from utils import rgb2lab, lab2rgb


# --------------------------------------------------------
# Argument Parsing
# --------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser('iColoriT Full-Resolution Inference', add_help=False)
    # Model
    parser.add_argument('--model_path', type=str, required=True, help='Path to your trained .pth checkpoint file.')
    parser.add_argument('--model', default='icolorit_base_4ch_patch16_224', type=str,
                        help='Name of model architecture.')

    # Input / Output
    parser.add_argument('--input_dir', type=str, required=True, help='Directory of full-resolution input images.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save colorized output images.')
    parser.add_argument('--mask_dir', type=str, default=None, help='(Optional) Directory of hint masks.')

    # Inference Parameters
    parser.add_argument('--patch_size', type=int, default=224, help='The size of patches the model was trained on.')
    parser.add_argument('--num_hints', type=int, default=10, help='Number of color hints to provide per patch.')
    parser.add_argument('--device', default='cuda', help='Device to use for inference.')

    return parser.parse_args()


# --------------------------------------------------------
# Helper Functions
# --------------------------------------------------------
def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        head_mode='cnn',
        use_rpb=True,
    )
    return model


def generate_hints_for_patch(mask_patch, num_hints):
    color_coords = np.argwhere(mask_patch)
    if num_hints == 0 or len(color_coords) == 0:
        return []

    num_hints_to_select = min(num_hints, len(color_coords))
    indices = np.random.choice(len(color_coords), num_hints_to_select, replace=False)
    selected_coords = color_coords[indices]

    return [(coord[1], coord[0]) for coord in selected_coords]


def detect_color_regions(image_patch_rgb, threshold=20):
    lab = cv2.cvtColor(image_patch_rgb, cv2.COLOR_RGB2LAB)
    a_channel = lab[:, :, 1].astype(np.float32)
    b_channel = lab[:, :, 2].astype(np.float32)
    color_magnitude = np.sqrt((a_channel - 128) ** 2 + (b_channel - 128) ** 2)
    return color_magnitude > threshold


def process_image(model, img_path, mask_path, args, device):
    try:
        img_pil = Image.open(img_path).convert('RGB')
        img_np = np.array(img_pil)
        H, W, _ = img_np.shape
    except Exception as e:
        print(f"Could not open image {img_path}. Skipping. Error: {e}")
        return

    full_hint_mask = None
    if mask_path and os.path.exists(mask_path):
        mask_pil = Image.open(mask_path).convert('L')
        if mask_pil.size != img_pil.size:
            mask_pil = mask_pil.resize(img_pil.size, Image.NEAREST)
        full_hint_mask = (np.array(mask_pil) > 127)

    output_ab = torch.zeros(1, 2, H, W).to(device)
    patch_counts = torch.zeros(1, 1, H, W).to(device)

    stride = args.patch_size // 2

    pbar_total = sum(1 for y in range(0, H, stride) for x in range(0, W, stride) if
                     y + args.patch_size <= H and x + args.patch_size <= W)
    pbar = tqdm(total=pbar_total, desc=f"Processing {os.path.basename(img_path)}")

    for y in range(0, H - args.patch_size + 1, stride):
        for x in range(0, W - args.patch_size + 1, stride):
            patch_rgb_np = img_np[y:y + args.patch_size, x:x + args.patch_size, :]
            patch_rgb_tensor = (torch.from_numpy(patch_rgb_np).permute(2, 0, 1) / 255.0).unsqueeze(0).to(device)
            patch_lab_tensor = rgb2lab(patch_rgb_tensor)

            if full_hint_mask is not None:
                mask_patch = full_hint_mask[y:y + args.patch_size, x:x + args.patch_size]
            else:
                mask_patch = detect_color_regions(patch_rgb_np)

            hint_coords = generate_hints_for_patch(mask_patch, args.num_hints)

            model_patch_size = model.patch_embed.patch_size[0]
            grid_dim = args.patch_size // model_patch_size
            bool_hint = torch.ones(1, grid_dim * grid_dim, dtype=torch.bool, device=device)
            for hx, hy in hint_coords:
                px = hx // model_patch_size
                py = hy // model_patch_size
                if 0 <= px < grid_dim and 0 <= py < grid_dim:
                    bool_hint[0, py * grid_dim + px] = False

            patch_mask_full_res = torch.zeros(1, 1, args.patch_size, args.patch_size, device=device)
            if hint_coords:
                hint_mask_patches = bool_hint.view(1, 1, grid_dim, grid_dim).float()
                patch_mask_full_res = F.interpolate(hint_mask_patches, size=(args.patch_size, args.patch_size),
                                                    mode='nearest')

            input_4ch = torch.cat([patch_lab_tensor, patch_mask_full_res], dim=1)

            with torch.no_grad():
                output_patch_3d = model(input_4ch, bool_hint)
                output_patch_ab = rearrange(output_patch_3d, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                                            h=grid_dim, w=grid_dim, p1=model_patch_size, p2=model_patch_size, c=2)

            output_ab[:, :, y:y + args.patch_size, x:x + args.patch_size] += output_patch_ab
            patch_counts[:, :, y:y + args.patch_size, x:x + args.patch_size] += 1
            pbar.update(1)

    pbar.close()

    output_ab_avg = output_ab / patch_counts.clamp(min=1)

    full_image_lab = rgb2lab(torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0).to(device)
    full_image_l = full_image_lab[:, :1, :, :]

    final_lab_image = torch.cat([full_image_l, output_ab_avg], dim=1)
    final_rgb_image = lab2rgb(final_lab_image)

    final_rgb_image_pil = Image.fromarray(
        (final_rgb_image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    save_path = osp.join(args.output_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_colorized.png")
    final_rgb_image_pil.save(save_path)
    print(f"Saved colorized image to: {save_path}")


# --------------------------------------------------------
# Main Execution
# --------------------------------------------------------
def main(args):
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    model = get_model(args)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {args.model_path}")

    print(f"Loading checkpoint from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    image_files = sorted(
        [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))])

    for img_file in image_files:
        img_path = os.path.join(args.input_dir, img_file)
        mask_path = None
        if args.mask_dir:
            # Try to find a matching mask file with common extensions
            base_name = os.path.splitext(img_file)[0]
            possible_masks = [f"{base_name}.png", f"{base_name}.jpg", f"{base_name}_mask.png"]
            for mask_name in possible_masks:
                if os.path.exists(os.path.join(args.mask_dir, mask_name)):
                    mask_path = os.path.join(args.mask_dir, mask_name)
                    break

        process_image(model, img_path, mask_path, args, device)


if __name__ == '__main__':
    args = get_args()
    main(args)