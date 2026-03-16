import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from skimage.color import rgb2lab

class PatchwiseCustomMaskHintDataset(Dataset):
    def __init__(self, image_folder, mask_folder, patch_size=224):
        self.patch_size = patch_size
        self.samples = []

        for fname in sorted(os.listdir(image_folder)):
            if not fname.lower().endswith('.jpg'):
                continue
            base = os.path.splitext(fname)[0]
            mask_path = os.path.join(mask_folder, base + '.png')
            img_path = os.path.join(image_folder, fname)

            if not os.path.exists(mask_path):
                continue

            img = cv2.imread(img_path)
            img = img[..., ::-1] / 255.0  # RGB

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = mask >= 1


            H, W, _ = img.shape
            for top in range(0, H - patch_size + 1, patch_size):
                for left in range(0, W - patch_size + 1, patch_size):
                    self.samples.append((img[top:top+patch_size, left:left+patch_size, :],
                                         mask[top:top+patch_size, left:left+patch_size],
                                         fname))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, mask, fname = self.samples[idx]

        lab = rgb2lab(img).astype(np.float32)
        L = lab[..., 0] / 100.0
        a = lab[..., 1] / 110.0
        b = lab[..., 2] / 110.0

        hint_ab = np.zeros((2, self.patch_size, self.patch_size), dtype=np.float32)
        hint_mask = np.zeros((1, self.patch_size, self.patch_size), dtype=np.float32)

        hint_ab[0][mask] = a[mask]
        hint_ab[1][mask] = b[mask]
        hint_mask[0][mask] = 1.0

        sample = {
            'L': torch.tensor(L).unsqueeze(0).float(),
            'ab': torch.tensor(np.stack([a, b], axis=0)).float(),
            'hint_ab': torch.tensor(hint_ab).float(),
            'hint_mask': torch.tensor(hint_mask).float(),
            'name': fname
        }
        return sample
