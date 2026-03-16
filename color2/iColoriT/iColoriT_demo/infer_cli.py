import torch
import numpy as np
import cv2
from PIL import Image
from skimage import color
from einops import rearrange
import timm
import modeling


# Load fine-tuned iColoriT model
def load_model(model_path, device='cuda'):
    model = timm.create_model('icolorit_base_4ch_patch16_224', pretrained=False,
                              use_rpb=True, avg_hint=True, head_mode='cnn', mask_cent=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device).eval()
    return model


# Normalize LAB input (-1 to 1)
def normalize_lab(lab_image):
    lab = lab_image.transpose((2, 0, 1))
    l_channel = (lab[[0]] - 50) / 100
    ab_channels = lab[1:] / 110
    return np.concatenate([l_channel, ab_channels], axis=0)


# Generate hint mask tensor
def prepare_hints(hint_positions_colors, img_lab):
    hint_img = np.zeros((224, 224, 3), dtype=np.float32)

    mask_channel = np.zeros((224, 224), dtype=np.float32) + 1.0  # default no-hint mask (1.0)

    for (x, y), rgb_color in hint_positions_colors:
        lab_color = color.rgb2lab(np.array([[rgb_color]], dtype=np.uint8))[0, 0, :]
        hint_img[y, x, 1:] = lab_color[1:] / 110  # ab channels normalized [-1,1]
        mask_channel[y, x] = 0  # mark hint location as 0 (hinted)

    hint_channels = np.zeros((3, 224, 224), dtype=np.float32)
    hint_channels[:2, :, :] = hint_img[:, :, 1:].transpose(2, 0, 1)
    hint_channels[2, :, :] = mask_channel  # Mask hint ~1 where no hint, 0 where hint exists

    return hint_channels


# Inference
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


# Example Usage
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model('/home/sapanagupta/MEGA/saligency_fps_thre_best_model/checkpoint-99.pth', device=device)

    # Define your expert hints: [(position), (RGB)]
    expert_hints = [
        ((80, 90), (0, 0, 0)),  # red at (x=50,y=100)
        ((15, 180), (0, 0, 0)),  # green at (x=100,y=150)
        ((115, 115), (0, 0, 0)),# blue at (x=150,y=200)
    ]

    # expert_hints = [
    #     ((50, 100), (255, 0, 0)),  # red at (x=50,y=100)
    #     ((100, 150), (0, 255, 0)),  # green at (x=100,y=150)
    #     ((150, 200), (0, 0, 255)),  # blue at (x=150,y=200)
    # ]

    colorize(
        model=model,
        gray_path='/home/sapanagupta/ICOLORIT_INPUTS/data/egypt/test/gray/egypt_88.jpg',
        hint_positions_colors=expert_hints,
        output_path='/home/sapanagupta/ICOLORIT_INPUTS/egypt_8black_new_colorized.png',
        device=device
    )