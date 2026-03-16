##!/bin/bash
#
## This script is for colorizing full-resolution images using a trained model.
## It uses the `infer_egyptian.py` script which handles patching, inferring, and stitching.
#
#export OMP_NUM_THREADS=1
#export CUDA_VISIBLE_DEVICES=${1:-0}
#
## --- 1. CONFIGURATION: UPDATE THESE PATHS ---
#
## Path to the specific model checkpoint you want to use.
## This should be one of the 'checkpoint-XX.pth' files from your training output directory.
#MODEL_CHECKPOINT="/home/sapanagupta/ICOLORIT_OUTPUTS/training_runs/exp_finetune_20250627_011404/checkpoint-4.pth" # <-- IMPORTANT: UPDATE THIS PATH
#
## Directory containing your FULL-RESOLUTION original Egyptian images to be colorized.
#INPUT_IMAGE_DIR="/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/data/Test/imgs_old/" # <-- UPDATE THIS IF DIFFERENT
#
## (Optional) Directory of full-resolution hint masks corresponding to the input images.
## If you don't provide this, hints will be auto-generated from the images themselves.
## Set to "" to disable.
#HINT_MASK_DIR="/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/data/Train/op/hints/test/" # <-- UPDATE OR SET TO ""
#
## Directory where the final, full-resolution colorized images will be saved.
## It's good practice to name the output after the model checkpoint used.
#OUTPUT_DIR="/home/sapanagupta/ICOLORIT_OUTPUTS/inference_results/$(basename "${MODEL_CHECKPOINT%.*}")"
#
#
## --- 2. SCRIPT EXECUTION ---
#
## Make sure the output directory exists
#mkdir -p "$OUTPUT_DIR"
#
#echo "=============================================="
#echo "Starting Full-Resolution Inference"
#echo "Model:        ${MODEL_CHECKPOINT}"
#echo "Input Images: ${INPUT_IMAGE_DIR}"
#if [ -n "$HINT_MASK_DIR" ]; then
#    echo "Hint Masks:   ${HINT_MASK_DIR}"
#fi
#echo "Output will be saved to: ${OUTPUT_DIR}"
#echo "=============================================="
#
#python /home/sapanagupta/PycharmProjects/color2/iColoriT/infer.py \
#    --model_path "$MODEL_CHECKPOINT" \
#    --input_dir "$INPUT_IMAGE_DIR" \
#    --output_dir "$OUTPUT_DIR" \
#    --mask_dir "$HINT_MASK_DIR" \
#    --num_hints 10 \
#    --patch_size 224
#
#echo "=============================================="
#echo "Inference complete."
#echo "Results saved in ${OUTPUT_DIR}"
#echo "=============================================="

####################################################################




#!/bin/bash

# This script is for colorizing full-resolution images using a trained model.
# It uses the `infer_egyptian.py` script which handles patching, inferring, and stitching.

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=${1:-0}

# --- 1. CONFIGURATION: UPDATE THESE PATHS ---

# Path to the specific model checkpoint you want to use.
# This should be one of the 'checkpoint-XX.pth' files from your training output directory.
MODEL_CHECKPOINT="/home/sapanagupta/ICOLORIT_OUTPUTS/Submission/output/icolorit_base_4ch_patch16_224/exp_20251005_185607/best_checkpoint.pth" # <-- IMPORTANT: UPDATE THIS PATH

# Directory containing your FULL-RESOLUTION original Egyptian images to be colorized.
INPUT_IMAGE_DIR="/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/INPUT_Thesis/Sapana/july_2nd_meet_work/Tast_patch_full_res_train/ICOLORIT_INPUTS/INPUTS/data/Test/" # <-- UPDATE THIS IF DIFFERENT

# (Optional) Directory of full-resolution hint masks corresponding to the input images.
# If you don't provide this, hints will be auto-generated from the images themselves.
# Set to "" to disable.
HINT_MASK_DIR="/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/INPUT_Thesis/Sapana/july_2nd_meet_work/Tast_patch_full_res_train/ICOLORIT_INPUTS/INPUTS/data/mini_op/hints/test/" # <-- UPDATE OR SET TO ""

# Directory where the final, full-resolution colorized images will be saved.
# It's good practice to name the output after the model checkpoint used.
OUTPUT_DIR="/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/output_Thesis/$(basename "${MODEL_CHECKPOINT%.*}")"


# --- 2. SCRIPT EXECUTION ---

# Make sure the output directory exists
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Starting Full-Resolution Inference"
echo "Model:        ${MODEL_CHECKPOINT}"
echo "Input Images: ${INPUT_IMAGE_DIR}"
if [ -n "$HINT_MASK_DIR" ]; then
    echo "Hint Masks:   ${HINT_MASK_DIR}"
fi
echo "Output will be saved to: ${OUTPUT_DIR}"
echo "=============================================="

python /home/sapanagupta/PycharmProjects/color2/iColoriT/infer.py \
    --model_path "$MODEL_CHECKPOINT" \
    --input_dir "$INPUT_IMAGE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --mask_dir "$HINT_MASK_DIR" \
    --num_hints 10 \
    --patch_size 224

echo "=============================================="
echo "Inference complete."
echo "Results saved in ${OUTPUT_DIR}"
echo "=============================================="