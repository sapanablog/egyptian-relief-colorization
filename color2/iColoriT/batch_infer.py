import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
from skimage.color import rgb2lab, lab2rgb


def prepare_patch(rgb_patch, mask_patch):
    lab = rgb2lab(rgb_patch).astype(np.float32)
    L = lab[..., 0] / 100.0
    a = lab[..., 1] / 110.0
    b = lab[..., 2] / 110.0

    hint_ab = np.zeros((2, 224, 224), dtype=np.float32)
    hint_mask = np.zeros((1, 224, 224), dtype=np.float32)

    hint_ab[0][mask_patch] = a[mask_patch]
    hint_ab[1][mask_patch] = b[mask_patch]
    hint_mask[0][mask_patch] = 1.0

    return (
        torch.tensor(L).unsqueeze(0).unsqueeze(0).float(),
        torch.tensor(hint_ab).unsqueeze(0).float(),
        torch.tensor(hint_mask).unsqueeze(0).float()
    )


def patchwise_inference(rgb, mask, model, device='cuda', patch_size=224, stride=112):
    H, W, _ = rgb.shape
    lab_full = rgb2lab(rgb).astype(np.float32)
    full_L = lab_full[..., 0] / 100.0

    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    rgb = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant')

    H_pad, W_pad = rgb.shape[:2]
    ab_pred = np.zeros((H_pad, W_pad, 2), dtype=np.float32)
    weight = np.zeros((H_pad, W_pad, 1), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for y in range(0, H_pad - patch_size + 1, stride):
            for x in range(0, W_pad - patch_size + 1, stride):
                rgb_patch = rgb[y:y + patch_size, x:x + patch_size]
                mask_patch = mask[y:y + patch_size, x:x + patch_size]

                L, hint_ab, hint_mask = prepare_patch(rgb_patch, mask_patch)
                L, hint_ab, hint_mask = L.to(device), hint_ab.to(device), hint_mask.to(device)

                out_ab = model(L, hint_ab, hint_mask)[0].cpu().numpy()
                out_ab = np.transpose(out_ab, (1, 2, 0))

                ab_pred[y:y + patch_size, x:x + patch_size] += out_ab
                weight[y:y + patch_size, x:x + patch_size] += 1.0

    ab_pred /= np.clip(weight, 1e-8, None)
    ab_pred = ab_pred[:H, :W, :]

    lab_merge = np.zeros((H, W, 3), dtype=np.float32)
    lab_merge[..., 0] = full_L[:H, :W] * 100.0
    lab_merge[..., 1:] = ab_pred * 110.0
    rgb_final = lab2rgb(lab_merge)

    return (rgb_final * 255).astype(np.uint8)


def batch_folder_processor(image_folder, mask_folder, output_folder, model, device='cuda'):
    os.makedirs(output_folder, exist_ok=True)
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

    for img_file in tqdm(image_files, desc="Batch Processing"):
        img_path = os.path.join(image_folder, img_file)
        mask_path = os.path.join(mask_folder, img_file)
        out_path = os.path.join(output_folder, img_file)

        if not os.path.exists(mask_path):
            print(f"❌ Mask not found for {img_file}, skipping.")
            continue

        rgb = cv2.imread(img_path)[..., ::-1] / 255.0
        mask = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127)

        result = patchwise_inference(rgb, mask, model, device=device)
        cv2.imwrite(out_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

