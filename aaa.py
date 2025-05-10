import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from lung_3d_dataset import Lung3DDataset
from unet_3d import UNet3D

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VAL_DIR = "data_3d/val"
MODEL_PATH = "checkpoints/unet3d_best.pth"
OUTPUT_DIR = "visualized_predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Modeli yükle
model = UNet3D(in_channels=1, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Val loader
val_dataset = Lung3DDataset(VAL_DIR)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# İlk 5 örneği tahmin et ve kaydet
with torch.no_grad():
    for idx, (x, y) in enumerate(val_loader):
        x = x.to(DEVICE)
        pred = torch.sigmoid(model(x)).cpu().numpy()[0, 0]
        img = x.cpu().numpy()[0, 0]
        gt = y.numpy()[0, 0]
        d = img.shape[0] // 2  # orta slice

        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        axs[0].imshow(img[d], cmap='gray')
        axs[0].set_title("Input")

        axs[1].imshow(gt[d], cmap='Greens')
        axs[1].set_title("Ground Truth")

        axs[2].imshow(pred[d], cmap='Reds')
        axs[2].set_title("Prediction")

        axs[3].imshow(img[d], cmap='gray')
        axs[3].imshow(gt[d], cmap='Greens', alpha=0.4)
        axs[3].imshow(pred[d] > 0.3, cmap='Reds', alpha=0.3)
        axs[3].set_title("Overlay")

        for ax in axs:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"val_pred_{idx}.png"), dpi=150)
        plt.close()

        if idx >= 4:
            break

print("✓ İlk 5 tahmin görselleştirildi → visualized_predictions klasörüne kaydedildi.")
