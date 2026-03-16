# import numpy as np
# import torch
# import cv2
# from skimage import color
# from einops import rearrange
# from PIL import Image
# import modeling  # Registers custom models
# from timm.models import create_model
# from scipy.ndimage import distance_transform_edt  # For propagation
#
# # Load model (direct instantiation)
# def load_model(model_path, device='cuda'):
#     from modeling import IColoriT
#     model = IColoriT(img_size=224, patch_size=16, in_chans=4, num_classes=512,
#                      embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
#                      qkv_bias=True, use_rpb=True, avg_hint=True,
#                      head_mode='cnn', mask_cent=False,
#                      init_values=0.)  # Fixed: Explicitly set to avoid NoneType error
#     checkpoint = torch.load(model_path, map_location=device)
#     model.load_state_dict(checkpoint['model'])
#     model.to(device).eval()
#     return model
#
# # Normalize LAB
# def normalize_lab(lab):
#     lab = lab.transpose((2, 0, 1))
#     l_channel = (lab[0] - 50) / 100
#     ab_channels = lab[1:] / 110
#     return np.concatenate([l_channel[np.newaxis], ab_channels], axis=0)
#
# # Prepare hints with larger patches
# def prepare_hints(hints_px, orig_shape, target_shape=(224, 224), patch_size=5):
#     scale_x = target_shape[1] / orig_shape[1]
#     scale_y = target_shape[0] / orig_shape[0]
#     hint_image = np.zeros(target_shape + (3,), dtype=np.float32)
#     hint_mask = np.ones(target_shape, dtype=np.float32)
#     colors = {"Dunkelblau": [0, 0, 139], "Hellblau": [173, 216, 230], "Rot": [255, 0, 0]}
#     for colorname, points in hints_px.items():
#         rgb = np.array(colors[colorname])
#         lab = color.rgb2lab(rgb.reshape(1,1,3)).squeeze()
#         for x, y in points:
#             x_s = int(x * scale_x)
#             y_s = int(y * scale_y)
#             x1, y1 = max(0, x_s - patch_size//2), max(0, y_s - patch_size//2)
#             x2, y2 = min(target_shape[1], x_s + patch_size//2 + 1), min(target_shape[0], y_s + patch_size//2 + 1)
#             hint_image[y1:y2, x1:x2, 1:] = lab[1:] / 110
#             hint_mask[y1:y2, x1:x2] = 0
#     hint_channels = np.concatenate([hint_image[:, :, 1:].transpose(2, 0, 1), hint_mask[np.newaxis]], axis=0)
#     return hint_channels
#
# # Low-res prediction
# def low_res_predict(model, img_path, hints_px, device):
#     orig = cv2.imread(img_path)[:, :, ::-1]
#     if orig is None:
#         raise ValueError(f"Failed to load image: {img_path}")
#     orig_shape = orig.shape[:2]
#     img_small = cv2.resize(orig, (224, 224))
#     img_small_lab = color.rgb2lab(img_small)
#     hint_tensor = torch.from_numpy(prepare_hints(hints_px, orig_shape)).unsqueeze(0).float().to(device)
#     input_tensor = torch.from_numpy(normalize_lab(img_small_lab)).unsqueeze(0).float().to(device)
#     with torch.no_grad():
#         ab_out = model(input_tensor, hint_tensor)
#     ab_out = rearrange(ab_out, 'b (h w) (p1 p2 c) -> b (h p1) (w p2) c', h=14, w=14, p1=16, p2=16)[0].cpu().numpy()
#     ab_out = np.tanh(ab_out) * 110
#     low_res_lab = np.dstack((img_small_lab[:, :, 0], ab_out))
#     low_res_rgb = (np.clip(color.lab2rgb(low_res_lab), 0, 1) * 255).astype(np.uint8)
#     print("Low-res prediction complete.")
#     return low_res_rgb, orig_shape
#
# # Sample and propagate colors to full-res
# def transfer_to_fullres(low_res_rgb, orig_path, hints_px, orig_shape):
#     orig = cv2.imread(orig_path)[:, :, ::-1]
#     if orig is None:
#         raise ValueError(f"Failed to load original image: {orig_path}")
#     orig_lab = color.rgb2lab(orig)
#     l_full = orig_lab[:, :, 0]
#     ab_full = np.zeros_like(orig_lab[:, :, 1:])
#     scale_x = 224 / orig_shape[1]
#     scale_y = 224 / orig_shape[0]
#     colors = {"Dunkelblau": [0, 0, 139], "Hellblau": [173, 216, 230], "Rot": [255, 0, 0]}
#     for colorname, points in hints_px.items():
#         for x, y in points:
#             x_low = min(max(int(x * scale_x), 0), 223)
#             y_low = min(max(int(y * scale_y), 0), 223)
#             sampled_ab = color.rgb2lab(low_res_rgb[y_low:y_low+1, x_low:x_low+1]).squeeze()[1:]
#             # Propagate (improved: use binary mask + distance for spread)
#             binary_mask = np.zeros_like(l_full, dtype=bool)
#             patch = 50  # Adjust spread radius
#             x1, y1 = max(0, x - patch), max(0, y - patch)
#             x2, y2 = min(orig_shape[1], x + patch), min(orig_shape[0], y + patch)
#             binary_mask[y1:y2, x1:x2] = True
#             dist_map = distance_transform_edt(~binary_mask)  # Distance from hint point
#             spread_mask = dist_map < patch
#             ab_full[spread_mask] = sampled_ab
#     lab_final = np.dstack((l_full, ab_full))
#     rgb_final = (np.clip(color.lab2rgb(lab_final), 0, 1) * 255).astype(np.uint8)
#     print("Full-res transfer complete.")
#     return rgb_final
#
# # Main
# if __name__ == "__main__":
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"Using device: {device}")
#     model = load_model('/home/sapanagupta/MEGA/saligency_fps_thre_best_model/checkpoint-99.pth', device)
#     hints_px = {
#         "Dunkelblau": [(3733, 1812), (2165, 1450), (3703, 1098), (3519, 1365), (2769, 1069), (1969, 2217), (1609, 1983),
#                        (1019, 2581)],
#         "Hellblau": [(2111, 201), (1555, 1391), (3511, 2745), (3457, 1677), (1387, 2447), (2653, 855), (1951, 919),
#                      (1091, 2145)],
#         "Rot": [(3275, 3339), (2553, 3335), (1555, 3357), (4001, 3309)]
#     }
#     low_res, orig_shape = low_res_predict(model, '/home/sapanagupta/ICOLORIT_INPUTS/custom_test/gray/egypt_1.jpg', hints_px, device)
#     full_res = transfer_to_fullres(low_res, '/home/sapanagupta/ICOLORIT_INPUTS/custom_test/old/I_2605.jpg', hints_px, orig_shape)
#     Image.fromarray(full_res).save('/home/sapanagupta/ICOLORIT_INPUTS/custom_test/old/full_colorized.png')
#     print('Done - check /home/sapanagupta/ICOLORIT_INPUTS/custom_test/old/full_colorized.png')

########################################
#
# import numpy as np
# import torch
# import cv2
# from skimage import color
# from einops import rearrange
# from PIL import Image
# import modeling  # Registers custom models
# from timm.models import create_model
# from scipy.ndimage import distance_transform_edt  # For propagation
#
# # Load model (direct instantiation)
# def load_model(model_path, device='cuda'):
#     from modeling import IColoriT
#     model = IColoriT(img_size=224, patch_size=16, in_chans=4, num_classes=512,
#                      embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
#                      qkv_bias=True, use_rpb=True, avg_hint=True,
#                      head_mode='cnn', mask_cent=False,
#                      init_values=0.)
#     checkpoint = torch.load(model_path, map_location=device)
#     model.load_state_dict(checkpoint['model'])
#     model.to(device).eval()
#     return model
#
# # Normalize LAB
# def normalize_lab(lab):
#     lab = lab.transpose((2, 0, 1))
#     l_channel = (lab[0] - 50) / 100
#     ab_channels = lab[1:] / 110
#     return np.concatenate([l_channel[np.newaxis], ab_channels], axis=0)
#
# # Prepare hints with larger patches
# def prepare_hints(hints_px, orig_shape, target_shape=(224, 224), patch_size=5):
#     scale_x = target_shape[1] / orig_shape[1]
#     scale_y = target_shape[0] / orig_shape[0]
#     hint_image = np.zeros(target_shape + (3,), dtype=np.float32)
#     hint_mask = np.ones(target_shape, dtype=np.float32)
#     colors = {"Dunkelblau": [0, 0, 139], "Hellblau": [173, 216, 230], "Rot": [255, 0, 0]}
#     for colorname, points in hints_px.items():
#         rgb = np.array(colors[colorname])
#         lab = color.rgb2lab(rgb.reshape(1,1,3)).squeeze()
#         for x, y in points:
#             x_s = int(x * scale_x)
#             y_s = int(y * scale_y)
#             x1, y1 = max(0, x_s - patch_size//2), max(0, y_s - patch_size//2)
#             x2, y2 = min(target_shape[1], x_s + patch_size//2 + 1), min(target_shape[0], y_s + patch_size//2 + 1)
#             hint_image[y1:y2, x1:x2, 1:] = lab[1:] / 110
#             hint_mask[y1:y2, x1:x2] = 0
#     hint_channels = np.concatenate([hint_image[:, :, 1:].transpose(2, 0, 1), hint_mask[np.newaxis]], axis=0)
#     return hint_channels
#
# # Low-res prediction
# def low_res_predict(model, img_path, hints_px, device):
#     orig = cv2.imread(img_path)[:, :, ::-1]
#     if orig is None:
#         raise ValueError(f"Failed to load image: {img_path}")
#     orig_shape = orig.shape[:2]
#     img_small = cv2.resize(orig, (224, 224))
#     img_small_lab = color.rgb2lab(img_small)
#     hint_tensor = torch.from_numpy(prepare_hints(hints_px, orig_shape)).unsqueeze(0).float().to(device)
#     input_tensor = torch.from_numpy(normalize_lab(img_small_lab)).unsqueeze(0).float().to(device)
#     with torch.no_grad():
#         ab_out = model(input_tensor, hint_tensor)
#     ab_out = rearrange(ab_out, 'b (h w) (p1 p2 c) -> b (h p1) (w p2) c', h=14, w=14, p1=16, p2=16)[0].cpu().numpy()
#     ab_out = np.tanh(ab_out) * 110
#     print(f"AB out min/max: {ab_out.min()} / {ab_out.max()}")  # Debug
#     low_res_lab = np.dstack((img_small_lab[:, :, 0], ab_out))
#     low_res_rgb = (np.clip(color.lab2rgb(low_res_lab), 0, 1) * 255).astype(np.uint8)
#     Image.fromarray(low_res_rgb).save('low_res_debug.png')  # Save for inspection
#     print("Low-res prediction complete. Check low_res_debug.png")
#     return low_res_rgb, orig_shape
#
# # Sample and propagate colors to full-res (fixed mask and sampling)
# def transfer_to_fullres(low_res_rgb, orig_path, hints_px, orig_shape):
#     orig = cv2.imread(orig_path)[:, :, ::-1]
#     if orig is None:
#         raise ValueError(f"Failed to load original image: {orig_path}")
#     orig_lab = color.rgb2lab(orig)
#     l_full = orig_lab[:, :, 0]
#     ab_full = np.zeros_like(orig_lab[:, :, 1:])
#     scale_x = 224 / orig_shape[1]
#     scale_y = 224 / orig_shape[0]
#     for colorname, points in hints_px.items():
#         for x, y in points:
#             x_low = min(max(int(x * scale_x), 0), 223)
#             y_low = min(max(int(y * scale_y), 0), 223)
#             # Sample from a small area for robustness
#             sample_area = low_res_rgb[max(0, y_low-2):y_low+3, max(0, x_low-2):x_low+3]
#             if sample_area.size == 0:
#                 continue
#             sampled_ab = np.mean(color.rgb2lab(sample_area / 255), axis=(0,1))[1:]
#             print(f"Sampled AB for {colorname} at ({x},{y}): {sampled_ab}")  # Debug
#             # Positive distance from hint point
#             hint_mask = np.zeros_like(l_full, dtype=bool)
#             patch = 100  # Increased for more spread
#             x1, y1 = max(0, x - patch), max(0, y - patch)
#             x2, y2 = min(orig_shape[1], x + patch), min(orig_shape[0], y + patch)
#             hint_mask[y1:y2, x1:x2] = True
#             dist_map = distance_transform_edt(hint_mask)  # Distance inside hint areas
#             spread_mask = dist_map < patch * 0.8  # Threshold for gradual spread
#             ab_full[spread_mask] = sampled_ab
#     lab_final = np.dstack((l_full, ab_full))
#     rgb_final = (np.clip(color.lab2rgb(lab_final), 0, 1) * 255).astype(np.uint8)
#     # Blend with original for better visibility (50% mix)
#     rgb_final = cv2.addWeighted(rgb_final, 0.7, orig, 0.3, 0)
#     print("Full-res transfer complete.")
#     return rgb_final
#
# # Main
# if __name__ == "__main__":
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"Using device: {device}")
#     model = load_model('/home/sapanagupta/ICOLORIT_INPUTS/MODELS/icolorit_base_4ch_patch16_224.pth', device)
#     hints_px = {
#         "Dunkelblau": [(3733, 1812), (2165, 1450), (3703, 1098), (3519, 1365), (2769, 1069), (1969, 2217), (1609, 1983),
#                        (1019, 2581)],
#         "Hellblau": [(2111, 201), (1555, 1391), (3511, 2745), (3457, 1677), (1387, 2447), (2653, 855), (1951, 919),
#                      (1091, 2145)],
#         "Rot": [(3275, 3339), (2553, 3335), (1555, 3357), (4001, 3309)]
#     }
#     low_res, orig_shape = low_res_predict(model, '/home/sapanagupta/ICOLORIT_INPUTS/custom_test/gray/egypt_1.jpg', hints_px, device)
#     full_res = transfer_to_fullres(low_res, '/home/sapanagupta/ICOLORIT_INPUTS/custom_test/old/I_2605.jpg', hints_px, orig_shape)
#     Image.fromarray(full_res).save('/home/sapanagupta/ICOLORIT_INPUTS/custom_test/old/full_colorized.png')
#     print('Done - check /home/sapanagupta/ICOLORIT_INPUTS/custom_test/old/full_colorized.png and low_res_debug.png for issues.')

#############################################


import numpy as np
import torch
import cv2
from skimage import color
from einops import rearrange
from PIL import Image
import modeling  # Registers custom models
from timm.models import create_model
from scipy.ndimage import distance_transform_edt  # For propagation

# Load model (direct instantiation)
def load_model(model_path, device='cuda'):
    from modeling import IColoriT
    model = IColoriT(img_size=224, patch_size=16, in_chans=4, num_classes=512,
                     embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                     qkv_bias=True, use_rpb=True, avg_hint=True,
                     head_mode='cnn', mask_cent=False,
                     init_values=0.)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device).eval()
    return model

# Normalize LAB
def normalize_lab(lab):
    lab = lab.transpose((2, 0, 1))
    l_channel = (lab[0] - 50) / 100
    ab_channels = lab[1:] / 110
    return np.concatenate([l_channel[np.newaxis], ab_channels], axis=0)

# Prepare hints with larger patches
def prepare_hints(hints_px, orig_shape, target_shape=(224, 224), patch_size=5):
    scale_x = target_shape[1] / orig_shape[1]
    scale_y = target_shape[0] / orig_shape[0]
    hint_image = np.zeros(target_shape + (3,), dtype=np.float32)
    hint_mask = np.ones(target_shape, dtype=np.float32)
    colors = {"Dunkelblau": [0, 0, 139], "Hellblau": [173, 216, 230], "Rot": [255, 0, 0]}
    for colorname, points in hints_px.items():
        rgb = np.array(colors[colorname])
        lab = color.rgb2lab(rgb.reshape(1,1,3)).squeeze()
        for x, y in points:
            x_s = int(x * scale_x)
            y_s = int(y * scale_y)
            x1, y1 = max(0, x_s - patch_size//2), max(0, y_s - patch_size//2)
            x2, y2 = min(target_shape[1], x_s + patch_size//2 + 1), min(target_shape[0], y_s + patch_size//2 + 1)
            hint_image[y1:y2, x1:x2, 1:] = lab[1:] / 110
            hint_mask[y1:y2, x1:x2] = 0
    hint_channels = np.concatenate([hint_image[:, :, 1:].transpose(2, 0, 1), hint_mask[np.newaxis]], axis=0)
    return hint_channels

# Low-res prediction
def low_res_predict(model, img_path, hints_px, device):
    orig = cv2.imread(img_path)[:, :, ::-1]
    if orig is None:
        raise ValueError(f"Failed to load image: {img_path}")
    orig_shape = orig.shape[:2]
    img_small = cv2.resize(orig, (224, 224))
    img_small_lab = color.rgb2lab(img_small)
    hint_tensor = torch.from_numpy(prepare_hints(hints_px, orig_shape)).unsqueeze(0).float().to(device)
    input_tensor = torch.from_numpy(normalize_lab(img_small_lab)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        ab_out = model(input_tensor, hint_tensor)
    ab_out = rearrange(ab_out, 'b (h w) (p1 p2 c) -> b (h p1) (w p2) c', h=14, w=14, p1=16, p2=16)[0].cpu().numpy()
    ab_out = np.tanh(ab_out) * 110
    print(f"AB out min/max: {ab_out.min()} / {ab_out.max()}")  # Debug
    low_res_lab = np.dstack((img_small_lab[:, :, 0], ab_out))
    low_res_rgb = (np.clip(color.lab2rgb(low_res_lab), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(low_res_rgb).save('low_res_debug.png')  # Save for inspection
    print("Low-res prediction complete. Check low_res_debug.png")
    return low_res_rgb, orig_shape


# Sample and propagate colors to full-res (fixed: upsample low-res AB to full size)
def transfer_to_fullres(low_res_rgb, orig_path, hints_px, orig_shape):
    orig = cv2.imread(orig_path)[:, :, ::-1]
    if orig is None:
        raise ValueError(f"Failed to load original image: {orig_path}")
    orig_lab = color.rgb2lab(orig)
    l_full = orig_lab[:, :, 0]  # Full-res L channel

    # Convert low_res_rgb back to LAB and extract AB
    low_res_lab = color.rgb2lab(low_res_rgb / 255.0)  # Normalize to [0,1] first
    ab_low = low_res_lab[:, :, 1:]  # Shape: (224, 224, 2)

    # Upsample AB to full resolution (bilinear for smoothness)
    ab_full = cv2.resize(ab_low, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR)

    # Combine with full-res L
    lab_final = np.dstack((l_full, ab_full))
    rgb_final = (np.clip(color.lab2rgb(lab_final), 0, 1) * 255).astype(np.uint8)

    print("Full-res transfer complete.")
    return rgb_final

# Main
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
# Sample and propagate colors to full-res (fixed mask and sampling)
def transfer_to_fullres(low_res_rgb, orig_path, hints_px, orig_shape):
    orig = cv2.imread(orig_path)[:, :, ::-1]
    if orig is None:
        raise ValueError(f"Failed to load original image: {orig_path}")
    orig_lab = color.rgb2lab(orig)
    l_full = orig_lab[:, :, 0]
    ab_full = np.zeros_like(orig_lab[:, :, 1:])
    scale_x = 224 / orig_shape[1]
    scale_y = 224 / orig_shape[0]
    for colorname, points in hints_px.items():
        for x, y in points:
            x_low = min(max(int(x * scale_x), 0), 223)
            y_low = min(max(int(y * scale_y), 0), 223)
            # Sample from a small area for robustness
            sample_area = low_res_rgb[max(0, y_low-2):y_low+3, max(0, x_low-2):x_low+3]
            if sample_area.size == 0:
                continue
            sampled_ab = np.mean(color.rgb2lab(sample_area / 255), axis=(0,1))[1:]
            print(f"Sampled AB for {colorname} at ({x},{y}): {sampled_ab}")  # Debug
            # Positive distance from hint point
            hint_mask = np.zeros_like(l_full, dtype=bool)
            patch = 100  # Increased for more spread
            x1, y1 = max(0, x - patch), max(0, y - patch)
            x2, y2 = min(orig_shape[1], x + patch), min(orig_shape[0], y + patch)
            hint_mask[y1:y2, x1:x2] = True
            dist_map = distance_transform_edt(hint_mask)  # Distance inside hint areas
            spread_mask = dist_map < patch * 0.8  # Threshold for gradual spread
            ab_full[spread_mask] = sampled_ab
    lab_final = np.dstack((l_full, ab_full))
    rgb_final = (np.clip(color.lab2rgb(lab_final), 0, 1) * 255).astype(np.uint8)
    # Blend with original for better visibility (50% mix)
    rgb_final = cv2.addWeighted(rgb_final, 0.7, orig, 0.3, 0)
    print("Full-res transfer complete.")
    return rgb_final
    model = load_model('/home/sapanagupta/ICOLORIT_INPUTS/MODELS/icolorit_base_4ch_patch16_224.pth', device)
    hints_px = {
        "Dunkelblau": [(3733, 1812), (2165, 1450), (3703, 1098), (3519, 1365), (2769, 1069), (1969, 2217), (1609, 1983),
                       (1019, 2581)],
        "Hellblau": [(2111, 201), (1555, 1391), (3511, 2745), (3457, 1677), (1387, 2447), (2653, 855), (1951, 919),
                     (1091, 2145)],
        "Rot": [(3275, 3339), (2553, 3335), (1555, 3357), (4001, 3309)]
    }
    low_res, orig_shape = low_res_predict(model, '/home/sapanagupta/ICOLORIT_INPUTS/custom_test/gray/egypt_1.jpg', hints_px, device)
    full_res = transfer_to_fullres(low_res, '/home/sapanagupta/ICOLORIT_INPUTS/custom_test/old/I_2605.jpg', hints_px, orig_shape)
    Image.fromarray(full_res).save('/home/sapanagupta/ICOLORIT_INPUTS/custom_test/old/full2_colorized.png')
    print('Done - check /home/sapanagupta/ICOLORIT_INPUTS/custom_test/old/full_colorized.png and low_res_debug.png for issues.')