# import os
#
# import argparse
# import numpy as np
# from PIL import Image
# import torch
# from torchvision import transforms
# from utils import rgb2lab, lab2rgb  # from original iColoriT repo
# from timm.models import create_model # from original iColoriT repo
#
#
# def read_hint_txt(path):
#     hints = []
#     with open(path, 'r') as f:
#         for line in f:
#             x, y, r, g, b = map(int, line.strip().split())
#             hints.append((x, y, r, g, b))
#     return hints
#
#
# def generate_hint_ab_mask(hints, H, W, device):
#     ab_hint = torch.zeros((1, 2, H, W), dtype=torch.float32, device=device)
#     hint_mask = torch.zeros((1, 1, H, W), dtype=torch.float32, device=device)
#
#     for x, y, r, g, b in hints:
#         pixel = torch.tensor([[[[r / 255.0]], [[g / 255.0]], [[b / 255.0]]]], device=device)  # [1, 3, 1, 1]
#         lab = rgb2lab(pixel).squeeze()  # [3]
#         a_val = lab[1].item()
#         b_val = lab[2].item()
#
#         if 0 <= x < W and 0 <= y < H:
#             ab_hint[0, 0, y, x] = a_val / 128.0
#             ab_hint[0, 1, y, x] = b_val / 128.0
#             hint_mask[0, 0, y, x] = 1.0
#     return ab_hint, hint_mask
#
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input', type=str, required=True, help='Path to grayscale PNG image')
#     parser.add_argument('--hint', type=str, required=True, help='Path to TXT file with x y R G B hints')
#     parser.add_argument('--output_dir', type=str, required=True)
#     parser.add_argument('--ckpt', type=str, required=True)
#     parser.add_argument('--model', type=str, default='icolorit_base_4ch_patch16_224')
#     args = parser.parse_args()
#
#     os.makedirs(args.output_dir, exist_ok=True)
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # Load grayscale and replicate to RGB
#     img_gray = Image.open(args.input).convert('L')
#     img_rgb = np.stack([np.array(img_gray)]*3, axis=-1) / 255.0
#     H, W = img_rgb.shape[:2]
#
#     hints = read_hint_txt(args.hint)
#
#     model = create_model(args.model, pretrained=False, head_mode='cnn', use_rpb=True)
#     model.load_state_dict(torch.load(args.ckpt, map_location=device)['model'])
#     model.to(device)
#     model.eval()
#
#     stride = 224
#     patch_size = 224
#     output = np.zeros((H, W, 3))
#     count_map = np.zeros((H, W, 1))
#
#     for top in range(0, H, stride):
#         for left in range(0, W, stride):
#             bottom = min(top + patch_size, H)
#             right = min(left + patch_size, W)
#             pad_h = patch_size - (bottom - top)
#             pad_w = patch_size - (right - left)
#
#             patch = img_rgb[top:bottom, left:right]
#             patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
#             patch_tensor = torch.tensor(patch.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
#
#             sub_hints = [(x - left, y - top, r, g, b) for x, y, r, g, b in hints
#                          if left <= x < right and top <= y < bottom]
#
#             ab_hint, hint_mask = generate_hint_ab_mask(sub_hints, patch_size, patch_size, device)
#
#             model_input = {
#                 "gray": patch_tensor[:, 0:1],
#                 "hint": ab_hint,
#                 "mask": hint_mask
#             }
#
#             with torch.no_grad():
#                 pred_ab = model(model_input).cpu().numpy()[0]
#
#             lab = np.concatenate([patch_tensor[:, 0:1].cpu().numpy()[0], pred_ab], axis=0)
#             pred_rgb = lab2rgb(torch.tensor(lab).unsqueeze(0)).squeeze().numpy()
#             pred_rgb = pred_rgb[:bottom-top, :right-left, :]
#
#             output[top:bottom, left:right] += pred_rgb
#             count_map[top:bottom, left:right] += 1
#
#     output /= np.clip(count_map, 1e-5, None)
#     out_name = os.path.splitext(os.path.basename(args.input))[0] + '_colorized.png'
#     Image.fromarray((output * 255).astype(np.uint8)).save(os.path.join(args.output_dir, out_name))
#
#
# if __name__ == '__main__':
#     main()
###########################################################################
# import os
# import argparse
# import numpy as np
# from PIL import Image
# import torch
# from utils import rgb2lab, lab2rgb  # from iColoriT repo
# import modeling  # must come before timm!
# from timm.models import create_model
#
# def read_hint_txt(path):
#     hints = []
#     with open(path, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if not parts:
#                 continue  # skip empty lines
#             x, y, r, g, b = map(int, parts)
#             hints.append((x, y, r, g, b))
#     return hints
#
# def generate_hint_ab_mask(hints, H, W, device):
#     ab_hint = torch.zeros((1, 2, H, W), dtype=torch.float32, device=device)
#     hint_mask = torch.zeros((1, 1, H, W), dtype=torch.float32, device=device)
#     for x, y, r, g, b in hints:
#         pixel = torch.tensor([[[[r / 255.0]], [[g / 255.0]], [[b / 255.0]]]], device=device)  # [1, 3, 1, 1]
#         lab = rgb2lab(pixel).squeeze()  # [3]
#         a_val = lab[1].item()
#         b_val = lab[2].item()
#         if 0 <= x < W and 0 <= y < H:
#             ab_hint[0, 0, y, x] = a_val / 128.0
#             ab_hint[0, 1, y, x] = b_val / 128.0
#             hint_mask[0, 0, y, x] = 1.0
#     return ab_hint, hint_mask
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input', type=str, required=True, help='Path to grayscale image')
#     parser.add_argument('--hint', type=str, required=True, help='Path to TXT file with x y R G B hints')
#     parser.add_argument('--output_dir', type=str, required=True)
#     parser.add_argument('--ckpt', type=str, required=True, help='Trained .pth checkpoint')
#     parser.add_argument('--model', type=str, default='icolorit_base_4ch_patch16_224')
#     parser.add_argument('--patch_size', type=int, default=224)
#     parser.add_argument('--stride', type=int, default=224)
#     parser.add_argument('--head_mode', type=str, default='cnn')
#     args = parser.parse_args()
#
#     os.makedirs(args.output_dir, exist_ok=True)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     img_gray = Image.open(args.input).convert('L')
#     img_rgb = np.stack([np.array(img_gray)]*3, axis=-1) / 255.0
#     H, W = img_rgb.shape[:2]
#
#     hints = read_hint_txt(args.hint)
#
#     # Import modeling before this!
#     model = create_model(args.model, pretrained=False, head_mode=args.head_mode, use_rpb=True)
#     model.load_state_dict(torch.load(args.ckpt, map_location=device)['model'])
#     model.to(device)
#     model.eval()
#
#     patch_size = args.patch_size
#     stride = args.stride
#     output = np.zeros((H, W, 3))
#     count_map = np.zeros((H, W, 1))
#
#     for top in range(0, H, stride):
#         for left in range(0, W, stride):
#             bottom = min(top + patch_size, H)
#             right = min(left + patch_size, W)
#             pad_h = patch_size - (bottom - top)
#             pad_w = patch_size - (right - left)
#
#             patch = img_rgb[top:bottom, left:right]
#             patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
#             patch_tensor = torch.tensor(patch.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
#
#             # Hints local to this patch
#             sub_hints = [(x - left, y - top, r, g, b) for x, y, r, g, b in hints
#                          if left <= x < right and top <= y < bottom]
#
#             ab_hint, hint_mask = generate_hint_ab_mask(sub_hints, patch_size, patch_size, device)
#             model_input = torch.cat([patch_tensor[:, 0:1], ab_hint, hint_mask], dim=1)  # [1, 4, patch_size, patch_size]
#
#             with torch.no_grad():
#                 pred_ab = model(model_input, mask=None).cpu().numpy()[0]  # [2, 224, 224]
#
#             # Compose Lab and convert to RGB
#             L = patch_tensor[:, 0:1].cpu().numpy()[0]
#             lab = np.concatenate([L, pred_ab], axis=0)
#             lab_tensor = torch.tensor(lab).unsqueeze(0)
#             pred_rgb = lab2rgb(lab_tensor).squeeze().cpu().numpy().transpose(1, 2, 0)
#             pred_rgb = pred_rgb[:bottom-top, :right-left, :]
#             output[top:bottom, left:right] += pred_rgb
#             count_map[top:bottom, left:right] += 1
#
#     output /= np.clip(count_map, 1e-5, None)
#     out_name = os.path.splitext(os.path.basename(args.input))[0] + '_colorized.png'
#     Image.fromarray(np.clip(output * 255, 0, 255).astype(np.uint8)).save(os.path.join(args.output_dir, out_name))
#     print("Saved:", os.path.join(args.output_dir, out_name))
#
# if __name__ == '__main__':
#     main()

#############################################################################



import os
from PIL import Image
import argparse
import numpy as np
import torch
from utils import rgb2lab, lab2rgb  # from iColoriT repo
import modeling  # must be imported before timm!
from timm.models import create_model
import torch.nn.functional as F

def read_hint_txt(path):
    hints = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            x, y, r, g, b = map(int, parts)
            hints.append((x, y, r, g, b))
    return hints

def generate_hint_ab_mask(hints, H, W, device):
    ab_hint = torch.zeros((1, 2, H, W), dtype=torch.float32, device=device)
    hint_mask = torch.zeros((1, 1, H, W), dtype=torch.float32, device=device)
    for x, y, r, g, b in hints:
        pixel = torch.tensor([[[[r / 255.0]], [[g / 255.0]], [[b / 255.0]]]], device=device)
        lab = rgb2lab(pixel).squeeze()
        a_val = lab[1].item()
        b_val = lab[2].item()
        if 0 <= x < W and 0 <= y < H:
            ab_hint[0, 0, y, x] = a_val / 128.0
            ab_hint[0, 1, y, x] = b_val / 128.0
            hint_mask[0, 0, y, x] = 1.0
    return ab_hint, hint_mask

def ensure_grayscale(img_path):
    img = Image.open(img_path)
    if img.mode != 'L':
        gray_img_path = os.path.splitext(img_path)[0] + "_gray.png"
        img_gray = img.convert('L')
        img_gray.save(gray_img_path)
        print(f"Converted {img_path} to grayscale and saved as {gray_img_path}")
        return gray_img_path
    else:
        print(f"{img_path} is already grayscale.")
        return img_path

def patch_seq_to_ab(pred_ab, patch_size, out_hw=16):
    """
    pred_ab: (1, 196, 512) (N, N_patch, 2*patch16*patch16)
    returns (2, patch_size, patch_size)
    """
    if pred_ab.shape[0] == 1:
        pred_ab = pred_ab[0]  # (196, 512)
    n_patches, feat = pred_ab.shape
    p = out_hw  # e.g. 16 for patch16
    assert feat == 2*p*p
    h_patch = w_patch = int(np.sqrt(n_patches))
    # [n_patches, 2*p*p] -> [h_patch, w_patch, 2, p, p]
    ab = pred_ab.reshape(h_patch, w_patch, 2, p, p)
    ab = np.transpose(ab, (2, 0, 3, 1, 4))  # [2, h_patch, p, w_patch, p]
    ab = ab.reshape(2, h_patch * p, w_patch * p)  # [2, patch_size, patch_size]
    return ab

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input (color or gray) image')
    parser.add_argument('--hint', type=str, required=True, help='TXT file with x y R G B hints')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--model', type=str, default='icolorit_base_4ch_patch16_224')
    parser.add_argument('--patch_size', type=int, default=224)
    parser.add_argument('--stride', type=int, default=224)
    parser.add_argument('--head_mode', type=str, default='cnn')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Step 1: Ensure grayscale input
    gray_img_path = ensure_grayscale(args.input)
    img_gray = Image.open(gray_img_path).convert('L')
    img_rgb = np.stack([np.array(img_gray)]*3, axis=-1) / 255.0
    H, W = img_rgb.shape[:2]

    hints = read_hint_txt(args.hint)

    model = create_model(args.model, pretrained=False, head_mode=args.head_mode, use_rpb=True)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)['model'])
    model.to(device)
    model.eval()

    patch_size = args.patch_size
    stride = args.stride
    output = np.zeros((H, W, 3))
    count_map = np.zeros((H, W, 1))

    # patch16 assumed; change out_hw=8 if using patch8 etc
    out_hw = 16 if patch_size == 224 else patch_size // (patch_size // 16)

    for top in range(0, H, stride):
        for left in range(0, W, stride):
            bottom = min(top + patch_size, H)
            right = min(left + patch_size, W)
            pad_h = patch_size - (bottom - top)
            pad_w = patch_size - (right - left)

            patch = img_rgb[top:bottom, left:right]
            patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
            patch_tensor = torch.tensor(patch.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

            sub_hints = [(x - left, y - top, r, g, b) for x, y, r, g, b in hints
                         if left <= x < right and top <= y < bottom]

            ab_hint, hint_mask = generate_hint_ab_mask(sub_hints, patch_size, patch_size, device)
            model_input = torch.cat([patch_tensor[:, 0:1], ab_hint, hint_mask], dim=1)

            with torch.no_grad():
                pred = model(model_input, mask=None).cpu().numpy()
            print("DEBUG: pred_ab.shape after model:", pred.shape)

            if pred.ndim == 3 and pred.shape[2] == 512 and pred.shape[1] == 196:
                # (1, 196, 512) typical for patch16
                ab = patch_seq_to_ab(pred, patch_size=patch_size, out_hw=16)
                print("DEBUG: ab.shape after seq2img:", ab.shape)
                pred_ab = ab
            elif pred.ndim == 2 and pred.shape[0] == 196 and pred.shape[1] == 512:
                # (196, 512)
                ab = patch_seq_to_ab(pred[None], patch_size=patch_size, out_hw=16)
                print("DEBUG: ab.shape after seq2img:", ab.shape)
                pred_ab = ab
            elif pred.ndim == 1:
                px = int(np.sqrt(pred.shape[0] // 2))
                pred_ab = pred.reshape(2, px, px)
            elif pred.ndim == 4:
                pred_ab = pred[0]
            elif pred.ndim == 3 and pred.shape[0] == 2:
                pred_ab = pred
            else:
                raise ValueError(f"Unrecognized pred_ab shape: {pred.shape}")

            if pred_ab.shape[1:] != (patch_size, patch_size):
                print(f"DEBUG: Upsampling pred_ab from {pred_ab.shape} to (2, {patch_size}, {patch_size})")
                pred_ab_torch = torch.tensor(pred_ab)[None]  # [1,2,H,W]
                pred_ab_upsampled = F.interpolate(pred_ab_torch, size=(patch_size, patch_size), mode='bilinear', align_corners=False)
                pred_ab = pred_ab_upsampled[0].cpu().numpy()
                print("DEBUG: pred_ab.shape after upsampling:", pred_ab.shape)

            L = patch_tensor[0, 0].cpu().numpy()[None, :, :]  # [1, patch_size, patch_size]
            lab = np.concatenate([L, pred_ab], axis=0)        # [3, patch_size, patch_size]
            lab_tensor = torch.tensor(lab).unsqueeze(0)
            pred_rgb = lab2rgb(lab_tensor).squeeze().cpu().numpy().transpose(1, 2, 0)
            pred_rgb = pred_rgb[:bottom-top, :right-left, :]
            output[top:bottom, left:right] += pred_rgb
            count_map[top:bottom, left:right] += 1

    output /= np.clip(count_map, 1e-5, None)
    base = os.path.splitext(os.path.basename(gray_img_path))[0]
    out_name = base + '_colorized.png'
    out_path = os.path.join(args.output_dir, out_name)
    Image.fromarray(np.clip(output * 255, 0, 255).astype(np.uint8)).save(out_path)
    print(f"\nColorized image saved as: {out_path}")

if __name__ == '__main__':
    main()
