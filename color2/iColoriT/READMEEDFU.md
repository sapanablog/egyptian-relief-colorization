# iColoriT – Egyptian Relief Colorization (Master’s Thesis)



Project: Colorization of Egyptian Relief Images

---

## Overview
This project fine-tunes the iColoriT (WACV 2023) Vision Transformer for colorizing ancient Egyptian temple relief images from the Edfu dataset.

Main contributions:
- Integration of mask-guided and saliency-based hint generation
- Structured hint levels (h2-n0 to h2-n20)
- Full-resolution inference pipeline
- Evaluation using PSNR, LPIPS, TPR, FPR, ROC, and AUC

---

## Environment Setup
1. Create the environment
   conda env create -f environment.yml
   conda activate color2

2. Confirm installation
   python --version
   torch --version

---

## Folder Structure
iColoriT/
├── iColoriT/                    core model and training code
├── preparation/Hints_Strategy/  hint generation scripts
├── scripts/                     training, inference, evaluation scripts
├── evaluation/                  evaluation metrics and tools
├── environment.yml              conda environment file
└── READMEedfu.md                    this file

---

## Hint Generation
1. Saliency method:
   python preparation/Hints_Strategy/make_mask_saliency_final.py \
     --input_dir /path/to/images \
     --output_dir /path/to/hints

2. Binary mask method:
   python preparation/Hints_Strategy/make_mask.py \
     --input_dir /path/to/images \
     --output_dir /path/to/masks

3. Structured hint sampling:
   python preparation/Hints_Strategy/ablation_hint_gen.py \
     --input_dir /path/to/masks \
     --output_dir /path/to/structured_hints

---

## Training
1. Using the shell script:
   bash scripts/train_thesis.sh

2. Or run manually:
   python iColoriT/train.py \
     --data_path /path/to/train \
     --val_data_path /path/to/val \
     --val_hint_dir /path/to/hints \
     --output_dir /path/to/output \
     --epochs 30 --batch_size 128

Model checkpoints are saved automatically to the output directory.

---

## Inference
1. Using the inference script:
   bash scripts/infer_full_thesis.sh

2. Or run manually:
   python iColoriT/infer.py \
     --input /path/to/grayscale/images \
     --hint_dir /path/to/hints \
     --output /path/to/save/colorized

---

## Evaluation
To compute PSNR, LPIPS, TPR, FPR, and other metrics:
python evaluation/eval_full_res_thesis.py \
  --gt_dir /path/to/ground_truth \
  --pred_dir /path/to/predictions \
  --pred_suffix "_colorized.png" \
  --resize_pred_to_gt \
  --save_path ./results_summary.txt \
  --per_image_csv ./per_image_metrics.csv

Output files:
- results_summary.txt  → average metrics
- per_image_metrics.csv  → per-image metrics

---

## Dataset
Edfu Temple Relief Dataset
- Field and web-sourced images
- Resized or patched to 224x224
- Each sample: grayscale input, hint mask, and color ground truth

---

## Example Workflow
conda env create -f environment.yml
conda activate color2

python preparation/Hints_Strategy/make_mask_saliency_final.py \
  --input_dir ./data/train/images \
  --output_dir ./data/train/hints/


bash scripts/train_thesis.sh
bash scripts/infer_full_thesis.sh

python evaluation/eval_full_res_thesis.py \
  --gt_dir ./data/Test_GT \
  --pred_dir ./output/results/

---

## Main Scripts
scripts/train_thesis.sh : training pipeline
scripts/infer_full_thesis.sh : inference on full-resolution images
evaluation/eval_full_res_thesis.py : evaluation metrics
preparation/Hints_Strategy/make_mask_saliency_final.py : saliency hint generation

---



Yun, J., Lee, S., Park, M., & Choo, J. (2023).
iColoriT: Towards Propagating Local Hints to the Right Region in Interactive Colorization by Leveraging Vision Transformer.
WACV 2023, pp. 1787–1796.

---

## Notes
- Tested on Ubuntu 22.04 with PyTorch 2.2, CUDA 12.1, and NVIDIA RTX 3090 GPU.
- All scripts assume relative paths as configured in train_thesis.sh and infer_full_thesis.sh.
- For evaluation, ensure predicted images and ground truth have matching filenames.

