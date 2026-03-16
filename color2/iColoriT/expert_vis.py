import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend BEFORE importing pyplot

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# File paths
img_input_path = "/home/sapanagupta/ICOLORIT_INPUTS/custom_test/egypt_6.jpg"
img_colorized_path = "/home/sapanagupta/ICOLORIT_INPUTS/custom_test/output_results/egypt_6_colorized_expert.png"

# Expert hint list (x, y, [R,G,B])
expert_hints = [
    (14857, 220, [255, 0, 0]),
    (10776, 1785, [255, 0, 0]),
    (8852, 921, [255, 0, 0]),
    (19807, 1801, [255, 0, 0]),
    (15017, 722, [0, 0, 139]),
    (15295, 636, [0, 0, 139]),
    (1470, 392, [0, 0, 139]),
    (8029, 679, [0, 0, 139]),
    (15296, 890, [135, 206, 250]),
    (15266, 518, [135, 206, 250]),
    (14761, 760, [135, 206, 250]),
    (5176, 726, [135, 206, 250]),
    (5576, 1091, [135, 206, 250]),
    (16364, 1689, [255, 255, 0]),
    (18731, 949, [255, 255, 0]),
    (20577, 919, [255, 255, 0]),
    (18312, 2330, [0, 0, 0]),
    (18542, 2832, [0, 0, 0]),
    (20765, 515, [0, 0, 0]),
    (19731, 1629, [0, 0, 0]),
]

# Load images
img_input = np.array(Image.open(img_input_path).convert("RGB"))
img_colorized = np.array(Image.open(img_colorized_path).convert("RGB"))

# Resize if needed for visualization
if img_input.shape != img_colorized.shape:
    img_colorized = cv2.resize(img_colorized, (img_input.shape[1], img_input.shape[0]))

# Overlay expert hints on the colorized output
overlay = img_colorized.copy()
for (x, y, color) in expert_hints:
    if x < overlay.shape[1] and y < overlay.shape[0]:
        cv2.circle(overlay, (x, y), 28, (0, 255, 0), 5)  # Green circle for hint

# Visualize and save to file
plt.figure(figsize=(18, 8))
plt.subplot(1, 2, 1)
plt.imshow(img_input)
plt.title("Original / Grayscale Input")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(overlay)
plt.title("Colorized Output + Hints Overlay")
plt.axis("off")

plt.tight_layout()
plt.savefig("visualization_output.png", dpi=200)
print("Visualization saved to visualization_output.png")

# Print color at each hint location
print("\nHint Locations and Color Check:")
for (x, y, expert_rgb) in expert_hints:
    if x < img_colorized.shape[1] and y < img_colorized.shape[0]:
        colorized_rgb = img_colorized[y, x]
        print(f"({x:5d}, {y:5d}): Output RGB = {colorized_rgb.tolist()}, Expert RGB = {expert_rgb}")
    else:
        print(f"({x:5d}, {y:5d}): Out of bounds for this image!")

# Compute metrics (between input and colorized)
psnr_score = psnr(img_input, img_colorized, data_range=255)
ssim_score = ssim(img_input, img_colorized, channel_axis=-1, data_range=255)
print(f"\nPSNR (Input vs Colorized): {psnr_score:.2f} dB")
print(f"SSIM (Input vs Colorized): {ssim_score:.4f}")
