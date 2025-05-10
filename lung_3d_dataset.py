import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class Lung3DDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_size=(128, 128)):
        self.data_dir = data_dir
        self.samples = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        self.transform = transform
        self.target_size = target_size  # (H, W)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = os.path.join(self.data_dir, self.samples[idx])
        data = np.load(sample_path)
        image = data['image']  # Shape: (D, H, W)
        mask = data['mask']    # Shape: (D, H, W)

        # Normalize
        image = image.astype(np.float32)
        mask = (mask > 0).astype(np.float32)

        # Add channel dimension: (1, D, H, W)
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        # Convert to torch tensor
        image = torch.tensor(image)
        mask = torch.tensor(mask)

        # Resize spatial dimensions: (1, D, H, W) → (1, D, 128, 128)
        if self.target_size != (image.shape[-2], image.shape[-1]):
            image = F.interpolate(image, size=self.target_size, mode='bilinear', align_corners=False)
            mask = F.interpolate(mask, size=self.target_size, mode='nearest')

        return image, mask
