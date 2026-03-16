import os
import torch
from torchvision.transforms import Compose, RandomResizedCrop, ToTensor, Normalize, Resize
import numpy as np
# Use your repo's dataset definitions
from dataset_folder_org import ImageFolder as RepoImageFolder
from dataset_folder_org import ImageWithFixedHint, ImageWithFixedHintAndCoord
from safe_dataset_wrapper import DropNoneWrapper
from adapter import coords_to_mask
import torchvision.transforms as T
# (Optional) torchvision ImageFolder if ever needed as a fallback
try:
    from torchvision.datasets import ImageFolder as TorchImageFolder
except Exception:
    TorchImageFolder = None


class DataAugmentationForIColoriT:
    def __init__(self, args):
        # No normalization on RGB space
        mean = [0., 0., 0.]
        std  = [1., 1., 1.]

        self.input_size = args.input_size
        self.model_patch_size = getattr(args, "model_patch_size", 16)

        self.transform = Compose([
            RandomResizedCrop(self.input_size),
            ToTensor(),
            Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        ])

        # ---- use PatchGridHint instead of the old RandomHintGenerator ----
        # args.num_hint_range can be (min, max) or a single int
        if isinstance(args.num_hint_range, (list, tuple)) and len(args.num_hint_range) == 2:
            num_min, num_max = int(args.num_hint_range[0]), int(args.num_hint_range[1])
        else:
            k = int(args.num_hint_range)
            num_min, num_max = k, k

        self.hint = PatchGridHint(
            input_size=self.input_size,
            patch_size=self.model_patch_size,   # 16 for 224/16 → 14×14 = 196
            num_hint_min=num_min,
            num_hint_max=num_max
        )

    def __call__(self, image):
        # return RGB tensor and a FLAT patch-grid mask [196]
        x = self.transform(image)                               # [3, 224, 224]
        h_vec = self.hint.sample_bool_vec()                     # [196] (True/False)
        # If your downstream expects 0/1 floats, uncomment:
        # h_vec = h_vec.float()
        return x, h_vec

    def __repr__(self):
        repr = "(DataAugmentationForIColoriT,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += f"  Hint generator = PatchGridHint(total={self.hint.total}, range=[{self.hint.num_hint_min}, {self.hint.num_hint_max}]),\n"
        repr += ")"
        return repr

class DataTransformationForIColoriT:
    def __init__(self, args):
        self.transform = Compose([
            Resize((args.input_size, args.input_size)),
            ToTensor(),
        ])

        from hint_generator import RandomHintGenerator, InteractiveHintGenerator
        if args.hint_generator == 'RandomHintGenerator':
            self.hint_generator = RandomHintGenerator(args.input_size, args.hint_size, args.num_hint_range)
        elif args.hint_generator == 'InteractiveHintGenerator':
            self.hint_generator = InteractiveHintGenerator(args.input_size, args.hint_size)
        else:
            raise NotImplementedError(f"{args.hint_generator} is not exist.")

    def __call__(self, image):
        return self.transform(image), self.hint_generator()

    def __repr__(self):
        repr = "(DataTransformationForIColoriT,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Hint generator = %s,\n" % str(self.hint_generator)
        repr += ")"
        return repr
####This is done for small dataset op only
class PatchGridHint:
    """
    Samples hints on the ViT patch grid (gh x gw), e.g. 14x14 for 224/16.
    Returns a boolean mask shaped [gh*gw] or [gh, gw] (True=hint present).
    """
    def __init__(self, input_size=224, patch_size=16, num_hint_min=0, num_hint_max=128):
        self.input_size = input_size
        self.patch_size = patch_size
        self.gh = input_size // patch_size
        self.gw = input_size // patch_size
        self.total = self.gh * self.gw
        self.num_hint_min = num_hint_min
        self.num_hint_max = num_hint_max
        print(f"Hint: total hint locations {self.total}, number of hints range "
              f"[{self.num_hint_min}, {self.num_hint_max}]")

    def sample_bool_vec(self, k=None):
        import torch
        if k is None:
            # uniform in [min, max]
            k = int(torch.randint(self.num_hint_min, self.num_hint_max + 1, (1,)).item())
        idx = torch.randperm(self.total)[:k]
        m = torch.ones(self.total, dtype=torch.bool)   # True = hint present
        m[idx] = False                                  # False = masked-out / no-hint if you prefer
        return m  # [196]

    def sample_bool_grid(self, k=None):
        m = self.sample_bool_vec(k).view(self.gh, self.gw)  # [gh, gw], e.g. [14, 14]
        return m
##till this much change for small dataset of patchgrid class
# class DataTransformationFixedHint:
#     def __init__(self, args) -> None:
#         self.input_size = args.input_size
#         self.hint_size = args.hint_size
#         self.img_transform = Compose([
#             Resize((self.input_size, self.input_size)),
#             ToTensor(),
#         ])
#         hint_dirs = args.hint_dirs
#         if isinstance(hint_dirs, str):
#             hint_dirs = [hint_dirs]
#         # hint subdir name convention: h{hint_size}-n{num}
#         self.num_hint = [int(os.path.basename(hd)[4:]) for hd in hint_dirs]
#         self.img_size = (self.input_size, self.input_size)
#         self.patch_size = 16  # for icolorit_base_4ch_patch16_224
#
#     def __call__(self, img, hint_coords):
#         return self.img_transform(img), self.coord2hint(hint_coords)


    # def coord2hint(self, hint_coords):
    #     H = self.input_size // self.hint_size
    #     hint = torch.ones((len(hint_coords), H, H))
    #     for idx, coords in enumerate(hint_coords):
    #         for x, y in coords:
    #             hint[idx, x // self.hint_size, y // self.hint_size] = 0
    #     return hint
# --- FIXED: compact, no stray methods, no indentation traps ---
class DataTransformationFixedHint:
    """
    Validation transform:
      - img_transform: torchvision transform (PIL -> tensor)
      - __call__(image, level_coords) -> (img_tensor, masks[L,196])
    """
    def __init__(self, img_transform=None, grid_dim=14, patch_size=16):
        self.img_transform = img_transform
        self.grid_dim = int(grid_dim)
        self.patch_size = int(patch_size)

    def __call__(self, image, level_coords):
        img_t = self.img_transform(image) if self.img_transform else T.ToTensor()(image)
        masks = [coords_to_mask(c, self.grid_dim, self.patch_size) for c in level_coords]
        masks = torch.stack(masks, dim=0)  # [L,196]
        return img_t, masks



    def coord2hint(self, coords):
        H, W = self.img_size
        ph = pw = self.patch_size
        gh, gw = H // ph, W // pw  # 14 x 14 -> 196

        def _zeros_vec():
            return torch.zeros((gh * gw,), dtype=torch.bool)

        def _coords_to_mask(c):
            if c is None:
                return _zeros_vec()
            if isinstance(c, torch.Tensor):
                c = c.detach().cpu().numpy()
            if np.isscalar(c):
                return _zeros_vec()

            c = np.array(c, dtype=float, copy=False)
            if c.size == 0:
                return _zeros_vec()

            # normalize to (K,2)
            if c.ndim == 1:
                if c.shape[0] == 2:
                    c = c.reshape(1, 2)
                else:
                    if c.shape[0] % 2 != 0:
                        c = c[: (c.shape[0] // 2) * 2]
                    c = c.reshape(-1, 2)
            else:
                c = c[..., :2].reshape(-1, 2)

            mask = np.zeros((gh, gw), dtype=bool)
            sx = W / gw
            sy = H / gh
            for x, y in c:
                x = float(np.clip(x, 0, W - 1))
                y = float(np.clip(y, 0, H - 1))
                gx = int(np.floor(x / sx))
                gy = int(np.floor(y / sy))
                mask[gy, gx] = True
            return torch.from_numpy(mask.reshape(-1))  # [196]

        # If coords is list-of-lists (levels), stack -> [L, 196]
        if isinstance(coords, (list, tuple)) and len(coords) > 0 and \
                isinstance(coords[0], (list, tuple, np.ndarray, torch.Tensor)):
            masks = [_coords_to_mask(c) for c in coords]
            return torch.stack(masks, dim=0)

        return _coords_to_mask(coords)

    def __repr__(self):
        repr = "(DataTransformationFixedHint,\n"
        repr += "  img_transform = %s,\n" % str(self.img_transform)
        repr += f"  Hint generator = Fixed, {self.num_hint}\n"
        repr += ")"
        return repr


class DataTransformationFixedHintContinuousCoords:
    def __init__(self, args) -> None:
        self.input_size = args.input_size
        self.hint_size = args.hint_size
        self.img_transform = Compose([
            Resize((self.input_size, self.input_size)),
            ToTensor(),
        ])
        hint_dirs = args.hint_dirs
        if isinstance(hint_dirs, str):
            hint_dirs = [hint_dirs]
        # names like "...:20" or "h2-n20" — we only need the last piece if colon used
        self.num_hint = [int(os.path.basename(hd).split(':')[-1].split('n')[-1]) for hd in hint_dirs]

    def __call__(self, img, hint_coords):
        # progressive lists of coords
        hint_coords = [hint_coords[0][:i] for i in range(len(hint_coords[0]) + 1)]
        return self.img_transform(img), self.coord2hint(hint_coords)

    def coord2hint(self, hint_coords):
        H = self.input_size // self.hint_size
        hint = torch.ones((len(hint_coords), H, H))
        for idx, coords in enumerate(hint_coords):
            for x, y in coords:
                hint[idx, x // self.hint_size, y // self.hint_size] = 0
        return hint

    def __repr__(self):
        repr = "(DataTransformationFixedHintContinuousCoords,\n"
        repr += "  img_transform = %s,\n" % str(self.img_transform)
        repr += f"  Hint generator = Fixed, {self.num_hint}\n"
        repr += ")"
        return repr


class DataTransformationFixedHintPrevCoods:
    def __init__(self, args) -> None:
        self.input_size = args.input_size
        self.hint_size = args.hint_size
        self.img_transform = Compose([
            Resize((self.input_size, self.input_size)),
            ToTensor(),
        ])
        hint_dirs = args.hint_dirs
        if isinstance(hint_dirs, str):
            hint_dirs = [hint_dirs]
        self.num_hint = [int(os.path.basename(hd).split(':')[-1].split('n')[-1]) for hd in hint_dirs]

    def __call__(self, img, hint_coords):
        pairs = [[coords[:-1], coords] for coords in hint_coords]
        return self.img_transform(img), self.coord2hint_prev(pairs)

    def coord2hint_prev(self, hint_coords):
        H = self.input_size // self.hint_size
        hint = torch.ones((len(hint_coords), 2, H, H))
        for idx, (prev_coords, coords) in enumerate(hint_coords):
            for x, y in prev_coords:
                hint[idx, 0, x // self.hint_size, y // self.hint_size] = 0
            for x, y in coords:
                hint[idx, 1, x // self.hint_size, y // self.hint_size] = 0
        return hint

    def __repr__(self):
        repr = "(DataTransformationFixedHintPrevCoods,\n"
        repr += "  img_transform = %s,\n" % str(self.img_transform)
        repr += f"  Hint generator = Fixed, {self.num_hint}\n"
        repr += ")"
        return repr


# =========================
# Dataset builder functions
# =========================

# def build_pretraining_dataset(args):
#     """TRAIN set.
#
#     If args.train_hint_base_dir and args.train_num_hint are set, use precomputed
#     patch-wise TXT hints via ImageWithFixedHint (root can be a flat folder of *.jpg).
#     Otherwise, fallback to repo ImageFolder (expects class subfolders).
#     """
#     transform = DataAugmentationForIColoriT(args)
#     print("Data Aug (TRAIN) = %s" % str(transform))
#
#     train_hints = getattr(args, "train_hint_base_dir", None)
#     num_hint   = getattr(args, "train_num_hint", None)
#
#     if train_hints and (num_hint is not None):
#         hint_dir = os.path.join(train_hints, f"h2-n{num_hint}")
#         if os.path.isdir(hint_dir):
#             # ImageWithFixedHint in your repo was patched to accept either root or root/imgs
#             return ImageWithFixedHint(
#                 args.data_path,
#                 [hint_dir],
#                 transform=transform,
#                 return_name=False,
#                 gray_file_list_txt=getattr(args, 'gray_file_list_txt', '')
#             )
#         else:
#             print(f"[WARN] Training hint dir not found: {hint_dir} — falling back to ImageFolder")
#
#     # Fallback (repo ImageFolder expects class subfolders; your train path is flat, so avoid unless needed)
#     if TorchImageFolder is not None:
#         return TorchImageFolder(root=args.data_path, transform=transform)
#     return RepoImageFolder(root=args.data_path, transform=transform)

##The above is for full dataset of op
##down is for small dataset op

def build_pretraining_dataset(args):
    transform = DataAugmentationForIColoriT(args)
    print("Data Aug (TRAIN) = %s" % str(transform))

    # Prefer repo ImageFolder (it works with flat folders & our (img, hint) transform)
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
    def _is_ok(p):
        ext = os.path.splitext(p)[1].lower()
        # drop torch cache and any weird files
        return ('.pt' not in p) and (ext in valid_exts)

    return RepoImageFolder(
        root=args.data_path,
        transform=transform,
        is_valid_file=_is_ok
    )


def build_validation_dataset(args):
    transform = DataTransformationForIColoriT(args)
    print("Data Trans (VAL) = %s" % str(transform))
    # Filter out cached tensors
    return RepoImageFolder(args.val_data_path, transform=transform,
                           is_valid_file=(lambda x: False if '.pt' in x else True))


# def build_fixed_validation_dataset(args):
#     transform = DataTransformationFixedHint(args)
#     print("Data Trans (VAL FIXED) = %s" % str(transform))
#     return ImageWithFixedHint(args.val_data_path, args.hint_dirs, transform=transform,
#                               return_name=args.return_name, gray_file_list_txt=args.gray_file_list_txt)
###this is done for validation check for small dataset


import os
import torchvision.transforms as T

# def build_fixed_validation_dataset(args):
#     # image transform for validation
#     val_tf = T.Compose([
#         T.Resize((args.input_size, args.input_size),
#                  interpolation=T.InterpolationMode.BILINEAR,
#                  antialias=True),
#         T.ToTensor(),
#     ])
#
#     # ---- construct the transformation wrapper correctly ----
#     # First try: (args, transform) as positional
#     try:
#         tf = DataTransformationFixedHint(args, val_tf)
#     except TypeError:
#         # Second try: (args, transform=...) as keyword
#         try:
#             tf = DataTransformationFixedHint(args, transform=val_tf)
#         except TypeError:
#             # Last resort: let the class build its own default transform from args
#             tf = DataTransformationFixedHint(args)
#     # --------------------------------------------------------
#
#     # Build the list of hint dirs (use normalized list from train.py if present)
#     hint_dirs = getattr(args, "hint_dirs", []) or [
#         os.path.join(args.val_hint_dir, f"h{args.hint_size}-n{int(n)}")
#         for n in args.val_hint_list
#     ]
#
#     ds = ImageWithFixedHint(
#         root=args.val_data_path,
#         hint_dirs=hint_dirs,
#         transform=tf,
#         return_name=getattr(args, "return_name", False),
#         gray_file_list_txt=getattr(args, "gray_file_list_txt", ""),
#         model_patch_size=args.patch_size[0],
#         input_size=args.input_size,
#     )
#     print(f"[VAL] will read hints from: {', '.join(ds.hint_dirs)}")
#     return ds

# def build_fixed_validation_dataset(args):
#     val_tf = T.Compose([T.Resize((args.input_size, args.input_size)), T.ToTensor()])
#     tf = DataTransformationFixedHint(
#         img_transform=val_tf,
#         grid_dim=args.input_size // args.patch_size[0],
#         patch_size=args.patch_size[0],
#     )
#

def build_fixed_validation_dataset(args):
    val_tf = T.Compose([
        T.Resize((args.input_size, args.input_size), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        T.ToTensor(),
    ])
    tf = DataTransformationFixedHint(
        img_transform=val_tf,
        grid_dim=args.input_size // args.patch_size[0],  # 224//16 = 14
        patch_size=args.patch_size[0],
    )

    hint_dirs = getattr(args, "hint_dirs", []) or [
        os.path.join(args.val_hint_dir, f"h{args.hint_size}-n{int(n)}") for n in args.val_hint_list
    ]

    ds = ImageWithFixedHint(
        root=args.val_data_path,
        hint_dirs=hint_dirs,
        transform=tf,
        return_name=getattr(args, "return_name", False),
        gray_file_list_txt=getattr(args, "gray_file_list_txt", ""),
        model_patch_size=args.patch_size[0],
        input_size=args.input_size,
    )
    print(f"[VAL] will read hints from: {', '.join(ds.hint_dirs)}")
    return ds


def build_fixed_validation_dataset_coord(args, without_tf=False):
    transform = DataTransformationFixedHintContinuousCoords(args) if not without_tf else None
    print("Data Trans (VAL COORD) = %s" % str(transform))
    return ImageWithFixedHintAndCoord(args.val_data_path, args.hint_dirs, transform=transform)


def build_fixed_validation_dataset_coord_2(args, without_tf=False):
    transform = DataTransformationFixedHintPrevCoods(args) if not without_tf else None
    print("Data Trans (VAL COORD PREV) = %s" % str(transform))
    return ImageWithFixedHintAndCoord(args.val_data_path, args.hint_dirs, transform=transform)
