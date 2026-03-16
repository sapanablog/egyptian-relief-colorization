# safe_dataset_wrapper.py
import torch

class DropNoneWrapper(torch.utils.data.Dataset):
    """
    Wrap any dataset whose __getitem__ may return None.
    We pre-scan once to keep only valid indices.
    """
    def __init__(self, base_ds, verbose=True, max_print=20):
        self.base = base_ds
        self.valid_idx = []
        bad = 0
        for i in range(len(base_ds)):
            try:
                sample = base_ds[i]
                if sample is None:
                    bad += 1
                else:
                    self.valid_idx.append(i)
            except Exception as e:
                bad += 1
                if verbose and bad <= max_print:
                    print(f"[VAL] Skipping idx={i} due to error: {e}")
        if verbose:
            print(f"[VAL] kept {len(self.valid_idx)}/{len(base_ds)} samples "
                  f"(dropped {bad}).")

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, j):
        return self.base[self.valid_idx[j]]
