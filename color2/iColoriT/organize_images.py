import os
import shutil
from glob import glob

def copy_predictions(src_dir, dest_dir, hint_dirs):
    """
    Copy the first 10 images from each subdirectory in the predictions source directory
    to the respective subdirectory in the destination directory.
    """
    os.makedirs(dest_dir, exist_ok=True)

    for hint_dir in hint_dirs:
        hint_src_dir = os.path.join(src_dir, hint_dir)
        hint_dest_dir = os.path.join(dest_dir, hint_dir)

        os.makedirs(hint_dest_dir, exist_ok=True)

        # List and copy the first 10 images
        image_files = sorted(glob(os.path.join(hint_src_dir, "*.png")))[:10]
        if not image_files:
            print(f"WARNING: No PNG files found in {hint_src_dir}")
            continue

        for file in image_files:
            shutil.copy(file, os.path.join(hint_dest_dir, os.path.basename(file)))

        print(f"Copied {len(image_files)} prediction images to {hint_dest_dir}")


def copy_groundtruth(src_dir, dest_dir, hint_dirs):
    """
    Copy the first 10 images from the flat ground truth directory into corresponding
    subdirectories matching the prediction structure.
    """
    os.makedirs(dest_dir, exist_ok=True)

    # List the first 10 images in the ground truth directory
    image_files = sorted(glob(os.path.join(src_dir, "*.jpeg")))[:10]
    if not image_files:
        print(f"WARNING: No JPEG files found in {src_dir}")
        return

    for hint_dir in hint_dirs:
        hint_dest_dir = os.path.join(dest_dir, hint_dir)
        os.makedirs(hint_dest_dir, exist_ok=True)

        for file in image_files:
            shutil.copy(file, os.path.join(hint_dest_dir, os.path.basename(file)))

        print(f"Copied {len(image_files)} ground truth images to {hint_dest_dir}")


if __name__ == "__main__":
    # Define source directories
    pred_dir = "/home/sapanagupta/PycharmProjects/color2/iColoriT/prediction"
    gt_dir = "/home/sapanagupta/PycharmProjects/color2/iColoriT/test"

    # Define destination directories for the organized images
    pred_dest_dir = "/home/sapanagupta/PycharmProjects/color2/iColoriT/organized_predictions"
    gt_dest_dir = "/home/sapanagupta/PycharmProjects/color2/iColoriT/organized_groundtruth"

    # Define the hint directories to process (only for predictions)
    hint_dirs = ["h2_n0", "h2_n1", "h2_n2", "h2_n5", "h2_n10", "h2_n20", "h2_n50", "h2_n100", "h2_n200"]

    # Process predictions
    print("Processing predictions...")
    copy_predictions(pred_dir, pred_dest_dir, hint_dirs)

    # Process ground truth
    print("Processing ground truth...")
    copy_groundtruth(gt_dir, gt_dest_dir, hint_dirs)
