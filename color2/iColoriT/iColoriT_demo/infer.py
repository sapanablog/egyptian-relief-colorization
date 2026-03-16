# import argparse
# import cv2
# import numpy as np
# import torch
# from skimage import color
# import modeling
# from timm.models import create_model
#
# def load_hints_txt(hints_path, img_size):
#     """
#     Load hints from txt: Each line x y r g b
#     Returns ab_img (2,H,W), mask (1,H,W)
#     """
#     ab_img = np.zeros((2, img_size, img_size), dtype=np.float32)
#     mask = np.zeros((1, img_size, img_size), dtype=np.uint8)
#     with open(hints_path, 'r') as f:
#         for line in f:
#             if line.strip() == '': continue
#             x, y, r, g, b = map(int, line.strip().split())
#             # Boundary check
#             if not (0 <= x < img_size and 0 <= y < img_size):
#                 print(f"Warning: hint ({x},{y}) out of bounds, skipping.")
#                 continue
#             lab = color.rgb2lab(np.array([[[r, g, b]]], dtype=np.uint8)).flatten()
#             ab_img[0, y, x] = lab[1]  # 'a' channel
#             ab_img[1, y, x] = lab[2]  # 'b' channel
#             mask[0, y, x] = 1
#     return ab_img, mask
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_path', type=str, required=True)
#     parser.add_argument('--input_image', type=str, required=True)
#     parser.add_argument('--hints_txt', type=str, required=True)
#     parser.add_argument('--output_image', type=str, required=True)
#     parser.add_argument('--model', type=str, default='icolorit_base_4ch_patch16_224')
#     parser.add_argument('--input_size', type=int, default=224)
#     parser.add_argument('--device', type=str, default='cuda')
#     args = parser.parse_args()
#
#     device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
#     img_size = args.input_size
#
#     # --- 1. Load Model ---
#     print("Loading model...")
#     model = create_model(args.model, pretrained=False)
#     checkpoint = torch.load(args.model_path, map_location='cpu')
#     model.load_state_dict(checkpoint['model'])
#     model.to(device)
#     model.eval()
#
#     # --- 2. Load and Preprocess Image ---
#     print("Loading image...")
#     img_bgr = cv2.imread(args.input_image)
#     if img_bgr is None:
#         raise ValueError(f"Failed to load image: {args.input_image}")
#
#     img_bgr = cv2.resize(img_bgr, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
#     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#     img_lab = color.rgb2lab(img_rgb)
#     im_l = img_lab[:, :, 0]
#     im_lab = img_lab.transpose((2,0,1))
#     # Normalization: L -> [-0.5, 0.5], ab -> [-1,1]
#     im_lab_norm = np.concatenate([[(im_lab[0]-50)/100], im_lab[1:]/110], axis=0)
#     im_lab_norm = torch.from_numpy(im_lab_norm).float().to(device)
#
#     # --- 3. Load Hints ---
#     print("Loading hints...")
#     ab_img, mask = load_hints_txt(args.hints_txt, img_size)
#     ab_mask = np.concatenate((ab_img / 110., (1-mask)), axis=0)
#     ab_mask = torch.from_numpy(ab_mask).float().to(device)
#
#     # --- 4. Inference ---
#     print("Running colorization inference...")
#     with torch.no_grad():
#         pred_ab = model(im_lab_norm.unsqueeze(0), ab_mask.unsqueeze(0))  # [1, N, 2]
#         # Reshape: [N, 2] -> [H, W, 2]
#         pred_ab = pred_ab.cpu().numpy()[0]
#         pred_ab = pred_ab.reshape(img_size, img_size, 2)
#         pred_ab = pred_ab * 110
#
#         # Recover LAB image for output
#         out_lab = np.zeros((img_size, img_size, 3), dtype=np.float32)
#         out_lab[:,:,0] = im_l  # original lightness
#         out_lab[:,:,1:] = pred_ab
#         out_rgb = color.lab2rgb(out_lab)
#         out_rgb = (np.clip(out_rgb, 0, 1) * 255).astype(np.uint8)
#
#     # --- 5. Save Output ---
#     cv2.imwrite(args.output_image, cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))
#     print(f"Saved colorized result to {args.output_image}")
#
# if __name__ == "__main__":
#     main()
#######################################################
# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import torch
from PIL import Image
from skimage.color import rgb2lab, gray2rgb
import modeling
from timm.models import create_model
import torchvision

def parse_args():
    parser = argparse.ArgumentParser(description='iColoriT Expert Hint Inference (GUI style)')
    parser.add_argument('--model_path', required=True, help='Path to checkpoint-xx.pth')
    parser.add_argument('--input_image', required=True, help='Grayscale input image (jpg or png)')
    parser.add_argument('--hints_txt', required=True, help='Expert hint txt (x y r g b per line)')
    parser.add_argument('--output_image', required=True, help='Where to save colorized result')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--model_name', default='icolorit_base_4ch_patch16_224',
                        help='Usually icolorit_base_4ch_patch16_224 or similar')
    return parser.parse_args()

def load_image(path):
    img = Image.open(path).convert('L').resize((224,224))
    arr = np.array(img).astype(np.float32) / 255.0
    return arr  # shape [224,224], values in 0~1

def load_hints(hints_txt, one_based=True):
    # expects each line: x y R G B
    coords = []
    with open(hints_txt) as f:
        for line in f:
            if line.strip() == '':
                continue
            x, y, r, g, b = map(int, line.strip().split())
            if one_based:    # Subtract 1 for 1-based hints
                x -= 1
                y -= 1
            coords.append((x, y, [r, g, b]))
    print(f"[DEBUG] Loaded {len(coords)} hints from {hints_txt}")
    return coords

def build_hint_mask(coords):
    # returns [1, 3, 224, 224] tensor in Lab, mask channel = 1 where hint placed
    hint_mask = np.zeros((3, 224, 224), dtype=np.float32)
    for x, y, rgb in coords:
        # Clamp coordinates (safety)
        x = min(max(x,0),223)
        y = min(max(y,0),223)
        # Convert RGB to Lab
        lab = rgb2lab(np.uint8([[rgb]]))[0,0]
        # Normalize ab to [-1, 1]
        a = lab[1] / 110.0
        b_ = lab[2] / 110.0
        hint_mask[0, y, x] = a  # a
        hint_mask[1, y, x] = b_  # b
        hint_mask[2, y, x] = 1.0     # mask present
        print(f"[DEBUG] Hint at ({x},{y}): RGB={rgb}, Lab={lab}, normed a,b=({a:.2f},{b_:.2f})")
    # Print number of nonzero mask entries
    print(f"[DEBUG] Hints placed at {np.count_nonzero(hint_mask[2])} locations.")
    return torch.from_numpy(hint_mask).unsqueeze(0)  # [1,3,224,224]

def prepare_input(gray_np):
    # Converts gray to Lab with fake a,b=0, shape [1,3,224,224], range matches training
    img_rgb = np.stack([gray_np]*3, axis=-1)
    lab = rgb2lab(img_rgb)          # shape [224,224,3], L in 0-100
    lab_tensor = np.zeros((1,3,224,224), dtype=np.float32)
    lab_tensor[0,0] = lab[:,:,0] / 50.0 - 1.0   # Normalize L to [-1,1]
    lab_tensor[0,1] = 0.0
    lab_tensor[0,2] = 0.0
    print(f"[DEBUG] Input L min/max: {lab_tensor[0,0].min():.2f}/{lab_tensor[0,0].max():.2f}")
    return torch.from_numpy(lab_tensor)

def save_lab_as_rgb(lab_tensor, save_path):
    # lab_tensor [1,3,224,224], values: L [-1,1], a,b = [-1,1] or 0
    lab_np = lab_tensor.squeeze(0).cpu().numpy()
    out_lab = np.zeros((224,224,3), dtype=np.float32)
    out_lab[:,:,0] = (lab_np[0]+1.0)*50.0   # L
    out_lab[:,:,1] = lab_np[1]*110.0        # a
    out_lab[:,:,2] = lab_np[2]*110.0        # b
    from skimage.color import lab2rgb
    out_rgb = lab2rgb(out_lab)
    out_img = (np.clip(out_rgb,0,1)*255).astype(np.uint8)
    Image.fromarray(out_img).save(save_path)
    print(f'Saved: {save_path}')

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # 1. Load model
    print(f"[DEBUG] Loading model {args.model_name} from {args.model_path}")
    model = create_model(args.model_name, pretrained=False, head_mode='cnn')
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'],strict=False)
    model = model.to(device).eval()

    # 2. Load grayscale image and hints
    gray_np = load_image(args.input_image)
    hint_coords = load_hints(args.hints_txt, one_based=True) # Set to False if your hint txt uses 0-based!
    hint_mask = build_hint_mask(hint_coords).to(device)

    # 3. Prepare input tensor [1,3,224,224]
    inp_lab = prepare_input(gray_np).to(device)

    # DEBUG visualize hint mask
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(inp_lab[0,0].cpu(), cmap='gray'); plt.title('Input L')
    plt.subplot(1,3,2); plt.imshow(hint_mask[0,0].cpu(), cmap='bwr', vmin=-1, vmax=1); plt.title('Hint a')
    plt.subplot(1,3,3); plt.imshow(hint_mask[0,2].cpu(), cmap='gray'); plt.title('Hint Mask')
    plt.tight_layout()
    plt.savefig('debug_hint_input.png')
    print("[DEBUG] Saved debug_hint_input.png showing L, a, and hint mask.")

    # 4. Run model (no grad, matches GUI)
    with torch.no_grad():
        print("[DEBUG] inp_lab shape:", inp_lab.shape)
        print("[DEBUG] hint_mask shape:", hint_mask.shape)
        print("[DEBUG] inp_lab stats: min %.3f max %.3f"%(inp_lab.min(), inp_lab.max()))
        print("[DEBUG] hint_mask stats: sum %.3f"%(hint_mask.sum()))
        pred_ab = model(inp_lab, hint_mask)  # Output shape [1,196,512] or similar
        print("[DEBUG] pred_ab shape:", pred_ab.shape)
        # GUI iColoriT: output = [B, N, 2*patch*patch], we need to reconstruct ab
        pred_ab = pred_ab.view(1, 196, 2, 16, 16)
        # Stitch back to [1,2,224,224]
        ab_out = torch.zeros(1,2,224,224, device=pred_ab.device)
        idx = 0
        for i in range(14):
            for j in range(14):
                ab_out[:,:,i*16:(i+1)*16, j*16:(j+1)*16] = pred_ab[0, idx]
                idx += 1
        # Full output [1,3,224,224]: merge L from input, ab from output
        out_lab = torch.zeros(1,3,224,224, device=ab_out.device)
        out_lab[:,0] = inp_lab[:,0]
        out_lab[:,1:] = ab_out
        # Save as RGB
        save_lab_as_rgb(out_lab.cpu(), args.output_image)

if __name__ == '__main__':
    main()
