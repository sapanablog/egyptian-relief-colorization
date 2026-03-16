import torch
import numpy as np
import cv2
from PIL import Image
from skimage import color
from einops import rearrange
import timm
import modeling   # Uncomment if you need a custom modeling import

##########################
# 1. Load iColoriT model #
##########################

def load_model(model_path, device='cuda'):
    model = timm.create_model('icolorit_base_4ch_patch16_224', pretrained=False,
                              use_rpb=True, avg_hint=True, head_mode='cnn', mask_cent=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device).eval()
    return model

###################################
# 2. Normalize LAB image for model #
###################################

def normalize_lab(lab_image):
    lab = lab_image.transpose((2, 0, 1))
    l_channel = (lab[[0]] - 50) / 100
    ab_channels = lab[1:] / 110
    return np.concatenate([l_channel, ab_channels], axis=0)

#########################################################
# 3. Prepare hints as tensor: list[((x,y), (R,G,B)),...] #
#########################################################

def prepare_hints(hint_positions_colors, img_lab):
    hint_img = np.zeros((224, 224, 3), dtype=np.float32)
    mask_channel = np.ones((224, 224), dtype=np.float32)  # default no-hint mask (1.0)

    for (x, y), rgb_color in hint_positions_colors:
        # RGB to Lab, scale as in model
        lab_color = color.rgb2lab(np.array([[rgb_color]], dtype=np.uint8))[0, 0, :]
        hint_img[y, x, 1:] = lab_color[1:] / 110  # ab normalized [-1,1]
        mask_channel[y, x] = 0  # mark hint location as 0 (hinted)

    hint_channels = np.zeros((3, 224, 224), dtype=np.float32)
    hint_channels[:2, :, :] = hint_img[:, :, 1:].transpose(2, 0, 1)
    hint_channels[2, :, :] = mask_channel
    return hint_channels

######################
# 4. Model Inference #
######################

def colorize(model, gray_path, hint_positions_colors, output_path, device='cuda'):
    img_gray = cv2.imread(gray_path)
    img_gray = cv2.resize(img_gray, (224, 224))
    img_lab = color.rgb2lab(img_gray[:, :, ::-1])
    input_tensor = torch.tensor(normalize_lab(img_lab)).unsqueeze(0).float().to(device)
    hint_tensor = torch.tensor(prepare_hints(hint_positions_colors, img_lab)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        ab = model(input_tensor, hint_tensor)

    ab = rearrange(ab, 'b (h w) (p1 p2 c) -> b (h p1) (w p2) c',
                   h=14, w=14, p1=16, p2=16)[0].cpu().numpy()

    lab_result = np.concatenate((img_lab[:, :, 0:1], ab * 110), axis=-1)
    rgb_result = (np.clip(color.lab2rgb(lab_result), 0, 1) * 255).astype(np.uint8)

    Image.fromarray(rgb_result).save(output_path)
    print(f"Colorized image saved to {output_path}")

############################
# 5. Coordinate Conversion #
############################

def scale_point(x, y, scale_x, scale_y):
    sx = int(round(x * scale_x))
    sy = int(round(y * scale_y))
    return min(max(sx, 0), 223), min(max(sy, 0), 223)

##################################
# 6. MAIN: Setup and Run Everything #
##################################

if __name__ == "__main__":
    # ---- Setup paths ----
    model_path = '/home/sapanagupta/ICOLORIT_INPUTS/data/egypt/output/icolorit_base_4ch_patch16_224/exp_250713_234105/checkpoint_epoch55.pth'
    full_res_image_path = "/home/sapanagupta/ICOLORIT_INPUTS/custom_test/I_2605.JPG"
    resized_image_path = "/home/sapanagupta/ICOLORIT_INPUTS/custom_test/I_2605_224.jpg"
    output_path = "/home/sapanagupta/ICOLORIT_INPUTS/custom_test/I_2605_colorized.png"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- Load and resize image ----
    full_img = Image.open(full_res_image_path)
    full_w, full_h = full_img.size
    scale_x, scale_y = 224 / full_w, 224 / full_h
    resized_img = full_img.resize((224, 224), Image.LANCZOS)
    resized_img.save(resized_image_path)

    # ---- Expert Hints (add/adjust RGB tuples as needed) ----
    # Use precise RGBs per expert guidance for Dunkelblau, Hellblau, Rot
    full_res_hints = [
        # Dunkelblau
        (3733, 1812, (17, 45, 109)),
        (2165, 1450, (17, 45, 109)),
        (3703, 1098, (17, 45, 109)),
        (3519, 1365, (17, 45, 109)),
        (2769, 1069, (17, 45, 109)),
        (1969, 2217, (17, 45, 109)),
        (1609, 1983, (17, 45, 109)),
        (1019, 2581, (17, 45, 109)),
        # Hellblau
        (2111, 201,  (89, 149, 227)),
        (1555, 1391, (89, 149, 227)),
        (3511, 2745, (89, 149, 227)),
        (3457, 1677, (89, 149, 227)),
        (1387, 2447, (89, 149, 227)),
        (2653, 855,  (89, 149, 227)),
        (1951, 919,  (89, 149, 227)),
        (1091, 2145, (89, 149, 227)),
        # Rot
        (3275, 3339, (184, 37, 32)),
        (2553, 3335, (184, 37, 32)),
        (1555, 3357, (184, 37, 32)),
        (4001, 3309, (184, 37, 32)),
    ]

    # ---- Scale expert hint coordinates to 224x224 ----
    expert_hints_224 = [
        (scale_point(x, y, scale_x, scale_y), rgb) for (x, y, rgb) in full_res_hints
    ]

    # ---- Load model ----
    model = load_model(model_path, device=device)

    # ---- Run colorization ----
    colorize(
        model=model,
        gray_path=resized_image_path,
        hint_positions_colors=expert_hints_224,
        output_path=output_path,
        device=device
    )
