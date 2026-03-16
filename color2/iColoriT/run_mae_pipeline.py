import os
from PIL import Image
import subprocess

# ---- SET THESE PATHS ----
color_img_path = "/home/sapanagupta/ICOLORIT_INPUTS/custom_test/egypt_6.jpg"         # Input color image
gray_img_path = "/home/sapanagupta/ICOLORIT_INPUTS/custom_test/egypt_6_gray.png"     # Will be saved as grayscale
hint_path = "/home/sapanagupta/ICOLORIT_INPUTS/custom_test/egypt_6.txt"              # Your expert hint txt file
output_dir = "/home/sapanagupta/ICOLORIT_INPUTS/custom_test/output"                  # Output dir for colorized
ckpt = "/home/sapanagupta/ICOLORIT_OUTPUTS/training_runs_4web_data_30Epoch/exp_finetune_20250628_020856/checkpoint-29.pth"  # Model checkpoint
run_mae_vis_path = "/home/sapanagupta/PycharmProjects/color2/iColoriT/run_mae_vis.py" # Path to your script

# 1. Convert color image to grayscale
img = Image.open(color_img_path).convert('L')
img.save(gray_img_path)
print(f"Grayscale image saved to: {gray_img_path}")

# 2. Run colorization inference with iColoriT
cmd = [
    "python", run_mae_vis_path,
    "--input", gray_img_path,
    "--hint", hint_path,
    "--output_dir", output_dir,
    "--ckpt", ckpt,
    "--head_mode", "cnn"
]
print("Running colorization inference...")
subprocess.run(cmd)

# 3. Print the output file location
out_name = os.path.splitext(os.path.basename(gray_img_path))[0] + '_colorized.png'
output_file = os.path.join(output_dir, out_name)
print(f"Colorized image should be saved at: {output_file}")
