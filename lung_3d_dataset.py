import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchio as tio  # ⬅️ Augmentasyonlar için

class Lung3DDataset(Dataset):
    def __init__(self, data_dir, augment=False, target_size=(128, 128)):
        self.data_dir = data_dir
        self.samples = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        self.augment = augment
        self.target_size = target_size  # (H, W)

        if self.augment:
            self.transform = tio.Compose([
                tio.RandomFlip(axes=(0,)),  # Z ekseninde flip
                tio.RandomAffine(scales=(0.9, 1.1), degrees=10),
                tio.RandomElasticDeformation(num_control_points=7, max_displacement=7),
                tio.RandomNoise(std=(0, 0.1)),
                tio.RandomGamma(log_gamma=(-0.3, 0.3))
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = os.path.join(self.data_dir, self.samples[idx])
        data = np.load(sample_path)
        image = data['image'].astype(np.float32)  # (D, H, W)
        mask = (data['mask'] > 0).astype(np.float32)  # (D, H, W)

        # Add channel dimension: (1, D, H, W)
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        # Convert to torch tensor
        image = torch.tensor(image)
        mask = torch.tensor(mask)

        # Resize to target size
        if self.target_size != (image.shape[-2], image.shape[-1]):
            image = F.interpolate(image, size=self.target_size, mode='bilinear', align_corners=False)
            mask = F.interpolate(mask, size=self.target_size, mode='nearest')

        # Apply augmentations
        if self.augment:
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=image),
                mask=tio.LabelMap(tensor=mask)
            )
            transformed = self.transform(subject)
            image = transformed.image.data
            mask = transformed.mask.data

        return image, mask
