import argparse, os, os.path as osp, sys, glob, csv, random, warnings
import numpy as np
import torch
from PIL import Image, ImageFile
from torchvision.transforms import functional as TF
from einops import rearrange
from tqdm import tqdm
import lpips

# trust your own images; silence Pillow warnings for >89MP
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

# --- import your utils.psnr (expects tensors in [0,1]) ---
CUR = osp.dirname(osp.abspath(__file__))
PAR = osp.dirname(CUR)
if PAR not in sys.path:
    sys.path.insert(0, PAR)
from utils import psnr as psnr_fn

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def list_images(d):
    return [f for f in sorted(os.listdir(d)) if osp.splitext(f)[1].lower() in IMG_EXTS]

def to01(pil_img):
    # PIL -> [1,3,H,W] in [0,1]
    return TF.to_tensor(pil_img).unsqueeze(0)

def to_m1p1(pil_img):
    # PIL -> [1,3,H,W] in [-1,1] (LPIPS requirement)
    t = TF.to_tensor(pil_img).unsqueeze(0)
    return t * 2.0 - 1.0

def pad_to_multiple(pil_img, m):
    w, h = pil_img.size
    nw = ((w + m - 1) // m) * m
    nh = ((h + m - 1) // m) * m
    if nw == w and nh == h:
        return pil_img
    return TF.pad(pil_img, [0, 0, nw - w, nh - h], fill=0)

def make_boundary_mask(patch_size, h, w, device):
    assert h % patch_size == 0 and w % patch_size == 0
    mask = torch.zeros((h, w), device=device)
    for i in range(h // patch_size - 1):
        mask[(i + 1) * patch_size - 1] = 1
        mask[(i + 1) * patch_size] = 1
    for j in range(w // patch_size - 1):
        mask[:, (j + 1) * patch_size - 1] = 1
        mask[:, (j + 1) * patch_size] = 1
    return mask

def boundary_psnr(img1_01, img2_01, patch=16, eps=1e-5):
    # img* : [1,3,H,W] in [0,1]
    _, _, H, W = img1_01.shape
    device = img1_01.device
    mask = make_boundary_mask(patch, H, W, device)  # [H,W]
    mse = torch.mean((img1_01 - img2_01) ** 2, dim=1)  # [1,H,W]
    mse = (mse * mask).sum(dim=(-1, -2)) / mask.sum()
    mse = torch.clamp(mse, min=eps)
    ps = 20 * torch.log10(torch.tensor(1.0, device=device) / torch.sqrt(mse))
    return ps.item()

def pev(img1_01, img2_01, patch=16):
    # img* : [1,3,H,W] in [0,1]
    mse = torch.mean((img1_01 * 255. - img2_01 * 255.) ** 2, dim=1)  # [1,H,W]
    H, W = mse.shape[-2:]
    assert H % patch == 0 and W % patch == 0
    tiles = rearrange(mse, 'b (h p1) (w p2) -> b (h w) (p1 p2)', p1=patch, p2=patch)
    rmse = torch.sqrt(tiles.mean(dim=2))  # [1, num_patches]
    return rmse.var(dim=1).mean().item()

def lpips_full_or_tiled(net, pred_m1p1, gt_m1p1, tile=False, tile_size=512, tile_pad=32, device="cuda"):
    """
    pred_m1p1, gt_m1p1 : [1,3,H,W] in [-1,1]
    If tile=True, compute LPIPS on overlapping tiles and average.
    """
    if not tile:
        return net(pred_m1p1.to(device), gt_m1p1.to(device)).item()

    _, _, H, W = pred_m1p1.shape
    s = tile_size
    p = tile_pad
    scores = []
    with torch.no_grad():
        for y in range(0, H, s):
            for x in range(0, W, s):
                y0 = max(y - p, 0)
                x0 = max(x - p, 0)
                y1 = min(y + s + p, H)
                x1 = min(x + s + p, W)
                pred_tile = pred_m1p1[:, :, y0:y1, x0:x1].to(device)
                gt_tile   = gt_m1p1[:, :, y0:y1, x0:x1].to(device)
                scores.append(net(pred_tile, gt_tile).item())
    return float(np.mean(scores)) if scores else net(pred_m1p1.to(device), gt_m1p1.to(device)).item()

def build_pairs(gt_dir, pred_dir, pred_suffix):
    gt_files = list_images(gt_dir)
    pred_files = list_images(pred_dir)
    pred_set = set(pred_files)
    pairs = []
    for g in gt_files:
        base = osp.splitext(g)[0]
        if pred_suffix:
            candidate = base + pred_suffix
            if candidate in pred_set:
                pairs.append((g, candidate))
            else:
                # try wildcard (e.g., different extension)
                hits = glob.glob(osp.join(pred_dir, base + pred_suffix.rsplit(".",1)[0] + ".*"))
                if hits:
                    pairs.append((g, osp.basename(hits[0])))
        else:
            # same basename
            found = None
            for p in pred_files:
                if osp.splitext(p)[0] == base:
                    found = p; break
            if found: pairs.append((g, found))
    if not pairs:
        raise RuntimeError("No GT/pred pairs matched. Check --pred_suffix.")
    return sorted(pairs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--pred_suffix", default="_colorized.png",
                    help="Suffix added to GT basename to locate prediction. Use '' if identical names.")
    ap.add_argument("--resize_pred_to_gt", action="store_true",
                    help="If set, resize prediction to GT size using bilinear.")
    ap.add_argument("--size", type=int, default=0,
                    help="If >0, resize BOTH images to this square size before metrics. "
                         "Use 0 to keep native resolution (recommended).")
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--lpips_backbone", choices=["alex","vgg","squeeze"], default="vgg")
    ap.add_argument("--tile", action="store_true", help="Enable tiled LPIPS to save memory.")
    ap.add_argument("--tile_size", type=int, default=512, help="Tile size for LPIPS.")
    ap.add_argument("--tile_pad", type=int, default=32, help="Overlap padding for LPIPS tiles.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_path", default="results/results_hr.txt")
    ap.add_argument("--per_image_csv", default="")
    args = ap.parse_args()

    random.seed(4885); np.random.seed(4885); torch.manual_seed(4885)

    pairs = build_pairs(args.gt_dir, args.pred_dir, args.pred_suffix)
    os.makedirs(osp.dirname(args.save_path), exist_ok=True)

    lpips_net = lpips.LPIPS(net=args.lpips_backbone).to(args.device).eval()

    tot = 0
    sum_psnr = sum_lpips = sum_bpsnr = sum_pev = 0.0
    writer = None
    if args.per_image_csv:
        os.makedirs(osp.dirname(args.per_image_csv), exist_ok=True)
        writer = csv.writer(open(args.per_image_csv, "w", newline=""))
        writer.writerow(["filename","psnr","lpips","boundary_psnr","pev"])

    pbar = tqdm(total=len(pairs), desc="Evaluating (HR)")
    with torch.no_grad():
        for gt_name, pred_name in pairs:
            gt_img   = Image.open(osp.join(args.gt_dir,  gt_name)).convert("RGB")
            pred_img = Image.open(osp.join(args.pred_dir, pred_name)).convert("RGB")

            # optional (rare size mismatch)
            if args.resize_pred_to_gt and pred_img.size != gt_img.size:
                pred_img = pred_img.resize(gt_img.size, Image.BILINEAR)

            # optional global resize (default 0 = native res)
            if args.size > 0:
                gt_eval   = gt_img.resize((args.size, args.size), Image.BILINEAR)
                pred_eval = pred_img.resize((args.size, args.size), Image.BILINEAR)
            else:
                gt_eval, pred_eval = gt_img, pred_img

            # tensors
            gt_01   = to01(gt_eval).to(args.device)
            pred_01 = to01(pred_eval).to(args.device)

            # PSNR at current eval size
            psnr_val = psnr_fn(pred_01, gt_01).item()

            # LPIPS at current eval size (full or tiled)
            gt_11   = (gt_01 * 2.0 - 1.0).to(args.device)
            pred_11 = (pred_01 * 2.0 - 1.0).to(args.device)
            lpips_val = lpips_full_or_tiled(
                lpips_net, pred_11, gt_11,
                tile=args.tile, tile_size=args.tile_size, tile_pad=args.tile_pad,
                device=args.device
            )

            # Boundary metrics require dims % patch_size == 0 → pad copies
            gt_pad   = pad_to_multiple(gt_eval,   args.patch_size)
            pred_pad = pad_to_multiple(pred_eval, args.patch_size)
            gt_pad_01   = to01(gt_pad).to(args.device)
            pred_pad_01 = to01(pred_pad).to(args.device)

            bpsnr_val = boundary_psnr(pred_pad_01, gt_pad_01, patch=args.patch_size)
            pev_val   = pev(pred_pad_01, gt_pad_01, patch=args.patch_size)

            sum_psnr  += psnr_val
            sum_lpips += lpips_val
            sum_bpsnr += bpsnr_val
            sum_pev   += pev_val
            tot += 1

            if writer:
                base = osp.splitext(gt_name)[0]
                writer.writerow([base, f"{psnr_val:.6f}", f"{lpips_val:.6f}",
                                 f"{bpsnr_val:.6f}", f"{pev_val:.6f}"])

            pbar.set_postfix({
                "PSNR": f"{sum_psnr/tot:.3f}",
                "LPIPS": f"{sum_lpips/tot:.5f}",
                "B-PSNR": f"{sum_bpsnr/tot:.3f}",
                "PEV": f"{sum_pev/tot:.3f}",
            })
            pbar.update(1)
    pbar.close()

    avg = {
        "psnr":          sum_psnr / tot,
        "lpips":         sum_lpips / tot,
        "boundary_psnr": sum_bpsnr / tot,
        "pev":           sum_pev / tot,
    }
    with open(args.save_path, "w") as f:
        f.write(f"total shown: {tot}\n{avg}\n")

    print("\n✅ Done.")
    print(f"Summary  -> {args.save_path}")
    if args.per_image_csv:
        print(f"Per-image -> {args.per_image_csv}")
    print(f"Averages -> PSNR: {avg['psnr']:.3f}, LPIPS({args.lpips_backbone}): {avg['lpips']:.5f}, "
          f"B-PSNR: {avg['boundary_psnr']:.3f}, PEV: {avg['pev']:.3f}")
if __name__ == "__main__":
    main()


####below is run this code
'''python /home/sapanagupta/PycharmProjects/color2/iColoriT/evaluation/eval_full_res_thesis.py \
  --gt_dir "/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/INPUT_Thesis/Sapana/july_2nd_meet_work/Tast_patch_full_res_train/ICOLORIT_INPUTS/INPUTS/data/Test" \
  --pred_dir "/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/output_Thesis/checkpoint-0" \
  --pred_suffix "_colorized.png" \
  --resize_pred_to_gt \
  --tile --tile_size 512 --tile_pad 32 \
  --save_path "/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/output_Thesis/checkpoint-0/results_summary.txt" \
  --per_image_csv "/home/sapanagupta/ICOLORIT_INPUTS/INPUTS/output_Thesis/checkpoint-0/per_image_metrics.csv"
'''