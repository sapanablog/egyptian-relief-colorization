import numpy as np
import torch
import timm
import cv2
from skimage import color
from einops import rearrange
from PIL import Image
import modeling

# Load your tuned iColoriT model from local modeling.py
def load_model(model_path, device='cuda'):
    model = modeling.icolorit_base_4ch_patch16_224(pretrained=False,
                                                   use_rpb=True,
                                                   avg_hint=True,
                                                   head_mode='cnn',
                                                   mask_cent=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device).eval()
    return model
# Normalize LAB input
def normalize_lab(lab):
    lab = lab.transpose((2, 0, 1))
    l_channel = (lab[[0]] - 50) / 100
    ab_channels = lab[1:] / 110
    return np.concatenate([l_channel, ab_channels], axis=0)


# Prepare scaled-down hints
def prepare_hints(hints_px, orig_shape, target_shape=(224, 224)):
    scale_x = target_shape[1] / orig_shape[1]
    scale_y = target_shape[0] / orig_shape[0]

    hint_image = np.zeros((target_shape[0], target_shape[1], 3), dtype=np.float32)
    hint_mask = np.ones((target_shape[0], target_shape[1]), dtype=np.float32)

    colors = {
        "Dunkelblau": [0, 0, 139],
        "Hellblau": [173, 216, 230],
        "Rot": [255, 0, 0]
    }

    for colorname, points in hints_px.items():
        rgb_value = np.array([colors[colorname]], dtype=np.uint8)
        lab_value = color.rgb2lab(rgb_value[None, :, :]).squeeze()
        for (x, y) in points:
            x_scaled = int(x * scale_x)
            y_scaled = int(y * scale_y)
            hint_image[y_scaled, x_scaled, 1:] = lab_value[1:] / 110.
            hint_mask[y_scaled, x_scaled] = 0

    hint_channels = np.zeros((3, target_shape[0], target_shape[1]), dtype=np.float32)
    hint_channels[:2, :, :] = hint_image[:, :, 1:].transpose(2, 0, 1)
    hint_channels[2, :, :] = hint_mask
    return hint_channels


# Colorization & upscaling Logic
def colorize_fullres(model, img_path, hints_px, device='cuda'):
    orig = cv2.imread(img_path)
    orig_lab = color.rgb2lab(orig[:, :, ::-1])

    # Prepare 224x224 image
    orig_shape = orig.shape[:2]
    img_small = cv2.resize(orig, (224, 224))
    img_small_lab = color.rgb2lab(img_small[:, :, ::-1])

    # Hint preparation
    hint_tensor = torch.tensor(prepare_hints(hints_px, orig_shape)).unsqueeze(0).float().to(device)

    # Normalize lab
    input_tensor = torch.tensor(normalize_lab(img_small_lab)).unsqueeze(0).float().to(device)

    # Model inference
    with torch.no_grad():
        ab_out = model(input_tensor, hint_tensor)

    ab_out = rearrange(ab_out, 'b (h w) (p1 p2 c) -> b (h p1) (w p2) c',
                       h=14, w=14, p1=16, p2=16)[0].cpu().numpy()

    # Upscale AB output
    ab_fullres = cv2.resize(ab_out, (orig.shape[1], orig.shape[0]))
    ab_fullres = ab_fullres * 110

    # Combine with Luminance
    lab_final = np.concatenate([orig_lab[:, :, 0:1], ab_fullres], axis=2)
    rgb_final = (np.clip(color.lab2rgb(lab_final), 0, 1) * 255).astype(np.uint8)

    return rgb_final


# Example Usage:
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model('/home/sapanagupta/ICOLORIT_OUTPUTS/training_runs_org_egptian_20epoch_with icoloritpretrained/exp_finetune_20250629_031737/checkpoint-19.pth', device=device)

    # Your expert hint positions for full-res image
    hints_px = {
        "Dunkelblau": [(3733, 1812), (2165, 1450), (3703, 1098), (3519, 1365), (2769, 1069), (1969, 2217), (1609, 1983),
                       (1019, 2581)],
        "Hellblau": [(2111, 201), (1555, 1391), (3511, 2745), (3457, 1677), (1387, 2447), (2653, 855), (1951, 919),
                     (1091, 2145)],
        "Rot": [(3275, 3339), (2553, 3335), (1555, 3357), (4001, 3309)]
    }

    output_img = colorize_fullres(model, '/home/sapanagupta/ICOLORIT_INPUTS/custom_test/old/I_2605.jpg', hints_px, device=device)
    Image.fromarray(output_img).save('/home/sapanagupta/ICOLORIT_INPUTS/custom_test/old/colorized_fullres_I19_2605.png')
    print('Saved full-resolution colorized image.')