import torch
from torch.utils.data import DataLoader
from lung_3d_dataset import Lung3DDataset
from unet_3d import UNet3D

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VAL_DIR = "data_3d/val"
MODEL_PATH = "checkpoints/unet3d_best.pth"

val_dataset = Lung3DDataset(VAL_DIR)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

model = UNet3D(in_channels=1, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def evaluate(model, loader, threshold):
    tp = fp = fn = tn = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = torch.sigmoid(model(x))
            pred_bin = (pred > threshold).float()
            y_bin = (y > 0.5).float()
            tp += (pred_bin * y_bin).sum().item()
            fp += (pred_bin * (1 - y_bin)).sum().item()
            fn += ((1 - pred_bin) * y_bin).sum().item()
            tn += ((1 - pred_bin) * (1 - y_bin)).sum().item()
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return threshold, dice, iou

print("Threshold\tDice\t\tIoU")
for t in [round(x, 2) for x in torch.linspace(0.1, 0.9, steps=9).tolist()]:
    th, dice, iou = evaluate(model, val_loader, threshold=t)
    print(f"{th:.2f}\t\t{dice:.4f}\t\t{iou:.4f}")
