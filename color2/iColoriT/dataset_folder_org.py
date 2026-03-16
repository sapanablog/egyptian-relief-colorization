# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------

import os
import os.path as osp
import random
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import torch
from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import torchvision.transforms as T  # ensure it's available at top-level

from adapter import coords_to_mask  # <-- make sure file is adapters.py

DEFAULT_INPUT_SIZE = 224
DEFAULT_MODEL_PATCH = 16

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


# --------------------- small helpers ---------------------

def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    return filename.lower().endswith(extensions)


def is_image_file(filename: str) -> bool:
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def pil_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def make_dataset(
    directory: str,
    class_to_idx: Dict[str, int],
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    instances = []
    directory = osp.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def _is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
        is_valid_file = _is_valid_file
    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = osp.join(directory, target_class)
        if not osp.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = osp.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    return instances


# --------------------- generic folders ---------------------

class DatasetFolder(VisionDataset):
    def __init__(
        self,
        root: str,
        loader: Callable[[str], Any],
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(DatasetFolder, self).__init__(root, transform=transform, target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = f"Found 0 files in subfolders of: {self.root}\n"
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        while True:
            try:
                path, target = self.samples[index]
                sample = self.loader(path)
                break
            except Exception as e:
                print(e)
                index = random.randint(0, len(self.samples) - 1)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


class ImageFolder(DatasetFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(ImageFolder, self).__init__(
            root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform, target_transform=target_transform,
            is_valid_file=is_valid_file
        )
        self.imgs = self.samples


# --------------------- iColoriT datasets ---------------------

# def _apply_two_or_one_arg_transform(transform, image, hint_coords_pixel, grid_dim, patch_size):
#     """
#     Try (image, coords) first; if the transform is image-only,
#     fall back to (image) and synthesize the hint mask.
#     Always returns (img_tensor, hint_mask_flat).
#     """
#     try:
#         out = transform(image, hint_coords_pixel)
#         if isinstance(out, tuple) and len(out) == 2:
#             img_t, hint_out = out
#             # accept (img_t, mask) OR (img_t, coords); if coords, synthesize mask:
#             if isinstance(hint_out, torch.Tensor) and hint_out.ndim == 1:
#                 return img_t, hint_out
#             else:
#                 hint_mask = coords_to_mask(hint_coords_pixel, grid_dim, patch_size)
#                 return img_t, hint_mask
#         else:
#             img_t = out
#             hint_mask = coords_to_mask(hint_coords_pixel, grid_dim, patch_size)
#             return img_t, hint_mask
#     except TypeError:
#         img_t = transform(image)
#         hint_mask = coords_to_mask(hint_coords_pixel, grid_dim, patch_size)
#         return img_t, hint_mask

import inspect
from typing import Any

def _apply_two_or_one_arg_transform(transform: Any, *args, **kwargs):
    """
    Robustly apply a transform that may accept either:
      - (image)                           # single-arg transforms / torchvision Compose
      - (image, hint_coords_pixel)        # our colorization transforms

    The dataset may pass extra positional args; we ignore them safely.
    """

    if not args:
        raise TypeError("_apply_two_or_one_arg_transform: missing 'image' argument")

    image = args[0]
    hint_coords_pixel = args[1] if len(args) >= 2 else None

    # Work out how many args the transform wants
    call = transform.__call__ if hasattr(transform, "__call__") else transform
    try:
        sig = inspect.signature(call)
        params = list(sig.parameters.values())

        # If it has *args or **kwargs, try (image, hint) first when we have hints
        has_var_pos = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
        has_var_kw  = any(p.kind == inspect.Parameter.VAR_KEYWORD   for p in params)

        # Count non-var positional params (exclude 'self')
        n_positional = sum(
            1 for p in params
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                          inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )
        if n_positional and params[0].name in ("self", "cls"):
            n_positional -= 1

        # Decision matrix
        if hint_coords_pixel is not None:
            # Prefer two-arg call if possible
            if has_var_pos or n_positional >= 2:
                return transform(image, hint_coords_pixel, **kwargs)
            elif n_positional >= 1:
                # Transform doesn’t accept hints; fall back to single-arg
                return transform(image, **kwargs)
            else:
                # No clear signature: try two-arg then one-arg
                try:
                    return transform(image, hint_coords_pixel, **kwargs)
                except TypeError:
                    return transform(image, **kwargs)
        else:
            # No hints available; just call single-arg
            return transform(image, **kwargs)

    except (ValueError, TypeError):
        # If we can't inspect the signature, optimistically try two-arg then one-arg
        if hint_coords_pixel is not None:
            try:
                return transform(image, hint_coords_pixel, **kwargs)
            except TypeError:
                return transform(image, **kwargs)
        return transform(image, **kwargs)


class ImageWithSpecificHint(Dataset):
    """
    TRAIN dataset: images in `images_dir`, hint coords in a single hint_dir (for a chosen n).
    Returns: ((img_tensor, hint_mask_flat), target)
    """
    def __init__(
        self,
        images_dir: str,
        hint_dir: str,
        transform=None,
        return_name: bool = False,
        gray_file_list_txt: str = '',
        model_patch_size: int = DEFAULT_MODEL_PATCH,
        input_size: int = DEFAULT_INPUT_SIZE,
    ):
        super().__init__()
        cand = osp.join(images_dir, 'imgs')
        self.images_dir = cand if osp.isdir(cand) else images_dir
        self.hint_dir = hint_dir
        self.transform = transform
        self.return_name = return_name

        self.model_patch_size = int(model_patch_size)
        self.input_size = int(input_size)
        self.grid_dim = self.input_size // self.model_patch_size

        if not osp.isdir(self.hint_dir):
            raise FileNotFoundError(f'{self.hint_dir} is not exist!')

        self.gray_imgs = []
        if gray_file_list_txt:
            with open(gray_file_list_txt, 'r') as f:
                self.gray_imgs = [osp.splitext(osp.basename(i))[0] for i in f.readlines()]

        self.img_list = sorted([
            f for f in os.listdir(self.images_dir)
            if is_image_file(f) and osp.splitext(f)[0] not in self.gray_imgs
        ])
        self.hint_list = sorted([
            f for f in os.listdir(self.hint_dir)
            if f.endswith('.txt') and osp.splitext(f)[0] not in self.gray_imgs
        ])

        if len(self.img_list) != len(self.hint_list):
            raise RuntimeError(f'Mismatch: images={len(self.img_list)} vs hints={len(self.hint_list)}')
        for img_f, hint_f in zip(self.img_list, self.hint_list):
            if osp.splitext(img_f)[0] != osp.splitext(hint_f)[0]:
                raise RuntimeError(f'Name mismatch: {img_f} vs {hint_f}')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_f = osp.join(self.images_dir, self.img_list[idx])
        img = Image.open(img_f).convert('RGB')

        hint_f = osp.join(self.hint_dir, self.hint_list[idx])
        with open(hint_f, 'r') as f:
            coords = [tuple(map(int, line.strip().split())) for line in f if line.strip()]

        # debug transform type once
        if not hasattr(self, "_printed_tf_once"):
            print("DEBUG dataset transform type:", type(self.transform))
            self._printed_tf_once = True

        if self.transform is not None:
            img_t, hint_mask_flat = _apply_two_or_one_arg_transform(
                self.transform, img, coords, self.grid_dim, self.model_patch_size
            )
        else:
            img_t = torchvision.transforms.ToTensor()(img)
            hint_mask_flat = coords_to_mask(coords, self.grid_dim, self.model_patch_size)

        target = 0
        if self.return_name:
            return (img_t, hint_mask_flat), target, self.img_list[idx]
        return (img_t, hint_mask_flat), target


class ImageWithFixedHint(Dataset):
    """
    VALIDATION/TEST dataset: supports one or multiple hint dirs (e.g., n in {0,1,2,5,10,20}).
    Returns: ((img_tensor, hint_mask_flat), target)
    """
    def __init__(
        self,
        root: str,
        hint_dirs,
        transform=None,
        return_name: bool = False,
        gray_file_list_txt: str = '',
        model_patch_size: int = DEFAULT_MODEL_PATCH,
        input_size: int = DEFAULT_INPUT_SIZE,
    ):
        super().__init__()
        cand = osp.join(root, 'imgs')
        self.img_dir = cand if osp.isdir(cand) else root

        if isinstance(hint_dirs, str):
            hint_dirs = [hint_dirs]
        for hd in hint_dirs:
            if not osp.isdir(hd):
                raise FileNotFoundError(f'{hd} is not exist!')

        self.hint_dirs = hint_dirs
        self.transform = transform
        self.return_name = return_name

        self.model_patch_size = int(model_patch_size)
        self.input_size = int(input_size)
        self.grid_dim = self.input_size // self.model_patch_size

        self.gray_imgs = []
        if gray_file_list_txt:
            with open(gray_file_list_txt, 'r') as f:
                self.gray_imgs = [osp.splitext(osp.basename(i))[0] for i in f.readlines()]

        self.img_list = sorted([
            f for f in os.listdir(self.img_dir)
            if is_image_file(f) and osp.splitext(f)[0] not in self.gray_imgs
        ])

        # per-dir hint lists + check stem equality
        self.hint_lists = []
        for hd in self.hint_dirs:
            hint_list = sorted([
                f for f in os.listdir(hd)
                if f.endswith('.txt') and osp.splitext(f)[0] not in self.gray_imgs
            ])
            if len(self.img_list) != len(hint_list):
                raise RuntimeError(f'Mismatch: images={len(self.img_list)} vs hints in {hd}={len(hint_list)}')
            for img_f, hint_f in zip(self.img_list, hint_list):
                if osp.splitext(img_f)[0] != osp.splitext(hint_f)[0]:
                    raise RuntimeError(f'Name mismatch: {img_f} vs {hint_f} in {hd}')
            self.hint_lists.append(hint_list)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_f = osp.join(self.img_dir, self.img_list[idx])
        img = Image.open(img_f).convert('RGB')

        # read coords for each hint level (one .txt per level)
        level_coords = []
        for hd, hint_list in zip(self.hint_dirs, self.hint_lists):
            hint_f = osp.join(hd, hint_list[idx])
            with open(hint_f, 'r') as f:
                coords = [tuple(map(int, line.strip().split())) for line in f if line.strip()]
            level_coords.append(coords)  # one list per level

        # Apply transform; returns (img_t, masks[L,196])
        if self.transform is not None:
            img_t, hint_mask_flat = _apply_two_or_one_arg_transform(
                self.transform, img, level_coords, self.grid_dim, self.model_patch_size
            )
        else:
            import torchvision.transforms as T
            img_t = T.ToTensor()(img)
            masks = [coords_to_mask(c, self.grid_dim, self.model_patch_size) for c in level_coords]
            hint_mask_flat = torch.stack(masks, dim=0)  # [L,196]

        target = 0
        if self.return_name:
            return (img_t, hint_mask_flat), target, self.img_list[idx]
        return (img_t, hint_mask_flat), target


class ImageWithFixedHintAndCoord(Dataset):
    """
    Validation/Test dataset variant that ALSO returns the raw hint coordinates.
    Returns: ((img_tensor, hint_mask_flat, raw_coords), target)
    """
    def __init__(
        self,
        root: str,
        hint_dirs,
        transform=None,
        return_name: bool = False,
        gray_file_list_txt: str = '',
        model_patch_size: int = DEFAULT_MODEL_PATCH,
        input_size: int = DEFAULT_INPUT_SIZE,
    ):
        super().__init__()
        cand = osp.join(root, 'imgs')
        self.img_dir = cand if osp.isdir(cand) else root

        if isinstance(hint_dirs, str):
            hint_dirs = [hint_dirs]
        for hd in hint_dirs:
            if not osp.isdir(hd):
                raise FileNotFoundError(f'{hd} is not exist!')

        self.hint_dirs = hint_dirs
        self.transform = transform
        self.return_name = return_name

        self.model_patch_size = int(model_patch_size)
        self.input_size = int(input_size)
        self.grid_dim = self.input_size // self.model_patch_size

        self.gray_imgs = []
        if gray_file_list_txt:
            with open(gray_file_list_txt, 'r') as f:
                self.gray_imgs = [osp.splitext(osp.basename(i))[0] for i in f.readlines()]

        self.img_list = sorted([
            f for f in os.listdir(self.img_dir)
            if is_image_file(f) and osp.splitext(f)[0] not in self.gray_imgs
        ])

        self.hint_lists = []
        for hd in self.hint_dirs:
            hint_list = sorted([
                f for f in os.listdir(hd)
                if f.endswith('.txt') and osp.splitext(f)[0] not in self.gray_imgs
            ])
            if len(self.img_list) != len(hint_list):
                raise RuntimeError(f'Mismatch: images={len(self.img_list)} vs hints in {hd}={len(hint_list)}')
            for img_f, hint_f in zip(self.img_list, hint_list):
                if osp.splitext(img_f)[0] != osp.splitext(hint_f)[0]:
                    raise RuntimeError(f'Name mismatch: {img_f} vs {hint_f} in {hd}')
            self.hint_lists.append(hint_list)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_f = osp.join(self.img_dir, self.img_list[idx])
        img = Image.open(img_f).convert('RGB')

        raw_coords = []
        for hd, hint_list in zip(self.hint_dirs, self.hint_lists):
            hint_f = osp.join(hd, hint_list[idx])
            with open(hint_f, 'r') as f:
                coords = [tuple(map(int, line.strip().split())) for line in f if line.strip()]
            raw_coords.extend(coords)

        if not hasattr(self, "_printed_tf_once"):
            print("DEBUG dataset transform type:", type(self.transform))
            self._printed_tf_once = True

        if self.transform is not None:
            try:
                out = self.transform(img, raw_coords)
                if isinstance(out, tuple) and len(out) == 2:
                    img_t, hint_out = out
                    if isinstance(hint_out, torch.Tensor) and hint_out.ndim == 1:
                        hint_mask_flat = hint_out
                    else:
                        hint_mask_flat = coords_to_mask(raw_coords, self.grid_dim, self.model_patch_size)
                else:
                    img_t = out
                    hint_mask_flat = coords_to_mask(raw_coords, self.grid_dim, self.model_patch_size)
            except TypeError:
                img_t = self.transform(img)
                hint_mask_flat = coords_to_mask(raw_coords, self.grid_dim, self.model_patch_size)
        else:
            img_t = T.ToTensor()(img)
            hint_mask_flat = coords_to_mask(raw_coords, self.grid_dim, self.model_patch_size)

        target = 0
        if self.return_name:
            return (img_t, hint_mask_flat, raw_coords), target, self.img_list[idx]
        return (img_t, hint_mask_flat, raw_coords), target
