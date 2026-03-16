# iColoriT/adapters.py
import torch

def coords_to_mask(hint_coords, grid_dim: int, patch_size: int):
    """
    Convert pixel (x,y) hint coords into a flattened boolean mask over the ViT patch grid.
    True = no hint; False = hint present at that patch.
    """
    mask = torch.ones((grid_dim, grid_dim), dtype=torch.bool)
    for x, y in hint_coords:
        px, py = x // patch_size, y // patch_size
        if 0 <= px < grid_dim and 0 <= py < grid_dim:
            mask[py, px] = False
    return mask.flatten()

def make_two_arg_transform(base_transform, grid_dim: int, patch_size: int):
    """
    Wrap an image-only transform so it accepts (image, hint_coords) and
    returns (image_tensor, hint_mask_flat).
    """
    def _wrapped(image, hint_coords):
        img_t = base_transform(image)          # image -> tensor
        hint_mask = coords_to_mask(hint_coords, grid_dim, patch_size)
        return img_t, hint_mask
    return _wrapped
