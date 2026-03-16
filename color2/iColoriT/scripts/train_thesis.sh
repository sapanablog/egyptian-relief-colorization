
#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=${1:-0}
MASTER_PORT=${2:-4885}

# --- Base Paths ---
PREPARED_DATA_ROOT="/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/INPUT_Thesis/Sapana/july_2nd_meet_work/Tast_patch_full_res_train/ICOLORIT_INPUTS/INPUTS/data/mini_op"

# --- Output ---
OUTPUT_BASE_DIR="/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/output_Thesis"
EXP_NAME="exp_finetune_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_BASE_DIR}/${EXP_NAME}"
LOG_DIR="${OUTPUT_DIR}/tf_logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# --- Pretrained ---
PRETRAINED_WEIGHTS="/home/sapanagupta/ICOLORIT_INPUTS/MODELS/icolorit_base_4ch_patch16_224.pth"

# --- Data paths ---
TRAIN_PATCH_DIR="${PREPARED_DATA_ROOT}/patches/train/"
VAL_IMG_DIR="${PREPARED_DATA_ROOT}/patches/val/imgs"         # <- point to the actual images folder
TRAIN_HINT_BASE_DIR="${PREPARED_DATA_ROOT}/hints/train"
VAL_HINT_BASE_DIR="${PREPARED_DATA_ROOT}/hints/val"

# build CSV of hint dirs that exist: h2-n0, h2-n1, h2-n2, h2-n5, h2-n10, h2-n20
VAL_LEVELS=(0 1 2 5 10 20)
VAL_HINT_DIRS=""
for n in "${VAL_LEVELS[@]}"; do
  d="${VAL_HINT_BASE_DIR}/h2-n${n}"
  if [[ -d "$d" ]]; then
    VAL_HINT_DIRS+="${d},"
  else
    echo "[WARN] Missing hint dir: $d"
  fi
done
VAL_HINT_DIRS="${VAL_HINT_DIRS%,}"  # strip trailing comma

echo "=============================================="
echo "Starting iColoriT Fine-Tuning"
echo "Training Patches Path: ${TRAIN_PATCH_DIR}"
echo "Validation Images Path: ${VAL_IMG_DIR}"
echo "Validation Hint Dirs:   ${VAL_HINT_DIRS}"
echo "Output will be saved to: ${OUTPUT_DIR}"
echo "=============================================="

torchrun --nproc_per_node=1 /home/sapanagupta/PycharmProjects/color2/iColoriT/train.py \
  --epochs 1 \
  --batch_size 128 \
  --num_workers 8 \
  --save_ckpt_freq 5 \
  --seed 4885 \
  --model icolorit_base_4ch_patch16_224 \
  --resume "${PRETRAINED_WEIGHTS}" \
  --resume_weights_only \
  --input_size 224 \
  --data_path "${TRAIN_PATCH_DIR}" \
  --val_data_path "${VAL_IMG_DIR}" \
  --hint_dirs "${VAL_HINT_DIRS}" \
  --gray_file_list_txt "" \
  --hint_size 2 \
  --train_hint_base_dir "${TRAIN_HINT_BASE_DIR}" \
  --train_num_hint 20 \
  --output_dir "${OUTPUT_DIR}" \
  --log_dir "${LOG_DIR}"
