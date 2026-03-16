import os
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
import lpips
from torch_fidelity import calculate_metrics
import numpy as np


def calculate_fid(gt_dir, pred_dir):
    """
    Calculate FID using torch-fidelity.
    """
    metrics = calculate_metrics(
        input1=pred_dir,
        input2=gt_dir,
        fid=True,
        verbose=False,
        samples_find_deep=True  # Allow subdirectories
    )
    return metrics['frechet_inception_distance']


def evaluate_metrics(gt_dir, pred_dirs, output_file):
    """
    Evaluate PSNR, SSIM, LPIPS, and FID for multiple predicted directories.
    """
    # LPIPS model
    lpips_model = lpips.LPIPS(net='alex')  # Use AlexNet for LPIPS

    results = []
    for pred_dir in pred_dirs:
        print(f"Evaluating {pred_dir}...")

        # Initialize accumulators for metrics
        psnr_values = []
        ssim_values = []
        lpips_values = []

        gt_images = os.listdir(gt_dir)
        total_images = len(gt_images)
        processed_images = 0

        for gt_image_name in gt_images:
            gt_image_path = os.path.join(gt_dir, gt_image_name)
            pred_image_path = os.path.join(pred_dir, gt_image_name)

            # Skip if prediction doesn't exist
            if not os.path.exists(pred_image_path):
                print(f"Warning: Missing prediction for {gt_image_name} in {pred_dir}")
                continue

            try:
                # Load images
                gt_image = lpips.im2tensor(lpips.load_image(gt_image_path))
                pred_image = lpips.im2tensor(lpips.load_image(pred_image_path))

                # PSNR
                psnr = compute_psnr(gt_image.numpy(), pred_image.numpy(), data_range=1.0)
                psnr_values.append(psnr)

                # SSIM
                ssim = compute_ssim(gt_image.numpy(), pred_image.numpy(), data_range=1.0, multichannel=True)
                ssim_values.append(ssim)

                # LPIPS
                lpips_value = lpips_model(gt_image, pred_image).item()
                lpips_values.append(lpips_value)

                processed_images += 1

            except Exception as e:
                print(f"Error processing {gt_image_name}: {e}")

        # Calculate averages
        avg_psnr = np.mean(psnr_values) if psnr_values else float('nan')
        avg_ssim = np.mean(ssim_values) if ssim_values else float('nan')
        avg_lpips = np.mean(lpips_values) if lpips_values else float('nan')

        # FID
        fid_value = calculate_fid(gt_dir, pred_dir)

        # Collect results
        results.append({
            'hint_level': os.path.basename(pred_dir),
            'psnr': avg_psnr,
            'ssim': avg_ssim,
            'lpips': avg_lpips,
            'fid': fid_value,
            'processed_images': processed_images,
            'total_images': total_images,
        })

        print(f"Results for {pred_dir}: PSNR={avg_psnr}, SSIM={avg_ssim}, LPIPS={avg_lpips}, FID={fid_value} "
              f"(Processed {processed_images}/{total_images} images)")

    # Save results to file
    with open(output_file, 'w') as f:
        for result in results:
            f.write(f"{result['hint_level']}: PSNR={result['psnr']}, SSIM={result['ssim']}, "
                    f"LPIPS={result['lpips']}, FID={result['fid']}, "
                    f"Processed {result['processed_images']}/{result['total_images']} images\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate metrics for iColoriT.')
    parser.add_argument('--gt_dir', required=True, help='Path to ground truth directory.')
    parser.add_argument('--pred_dir', required=True, help='Path to predicted images directory (with subdirectories).')
    parser.add_argument('--output_file', default='evaluation_results.txt', help='File to save results.')
    args = parser.parse_args()

    # List all subdirectories in pred_dir
    pred_subdirs = [os.path.join(args.pred_dir, d) for d in os.listdir(args.pred_dir) if os.path.isdir(os.path.join(args.pred_dir, d))]

    # Evaluate metrics
    evaluate_metrics(args.gt_dir, pred_subdirs, args.output_file)
