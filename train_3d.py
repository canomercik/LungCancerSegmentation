import os
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from lung_3d_dataset import Lung3DDataset
from unet_3d import UNet3D
from tqdm import tqdm
import matplotlib.pyplot as plt

# === Ayarlar ===
BATCH_SIZE = 1
NUM_EPOCHS = 35
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data_3d/train"
VAL_DIR = "data_3d/val"
SAVE_PATH = "checkpoints/unet3d_best.pth"
CSV_LOG = "results.csv"
PLOT_PATH = "training_curve.png"
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# === Eğitim Fonksiyonu ===
def train(model, loader, optimizer, loss_fn, scaler):
    model.train()
    loop = tqdm(loader, desc="Training")
    total_loss = 0.0

    for x, y in loop:
        x, y = x.to(DEVICE), y.to(DEVICE)

        with torch.amp.autocast(device_type='cuda', enabled=True):
            preds = model(x)
            loss = loss_fn(preds, y)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)

# === Değerlendirme Fonksiyonu ===
def evaluate(model, loader, threshold=0.3):
    model.eval()
    total_tp = total_fp = total_fn = total_tn = 0.0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = torch.sigmoid(model(x))
            preds_bin = (preds > threshold).float()
            y_bin = (y > 0.5).float()  # Soft label'i binary yap

            tp = (preds_bin * y_bin).sum().item()
            fp = (preds_bin * (1 - y_bin)).sum().item()
            fn = ((1 - preds_bin) * y_bin).sum().item()
            tn = ((1 - preds_bin) * (1 - y_bin)).sum().item()

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

    dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn + 1e-8)
    iou = total_tp / (total_tp + total_fp + total_fn + 1e-8)
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + 1e-8)
    specificity = total_tn / (total_tn + total_fp + 1e-8)

    model.train()

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "specificity": specificity,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "tn": total_tn,
    }

# === Ana Eğitim Döngüsü ===
def main():
    train_dataset = Lung3DDataset(TRAIN_DIR)
    val_dataset = Lung3DDataset(VAL_DIR)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = UNet3D(in_channels=1, out_channels=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()
    best_dice = 0.0

    with open(CSV_LOG, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Epoch", "Train Loss", "Dice", "IoU", "Precision", "Recall",
            "Accuracy", "Specificity", "TP", "FP", "FN", "TN", "Time (s)"
        ])

    epoch_losses = []

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        train_loss = train(model, train_loader, optimizer, loss_fn, scaler)
        metrics = evaluate(model, val_loader, threshold=0.3)

        print(f"\n📊 Validation Metrics:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"{k.capitalize():<12}: {v:.4f}")

        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"✓ En iyi model kaydedildi: {SAVE_PATH}")

        duration = time.time() - start_time
        epoch_losses.append((epoch+1, train_loss, metrics["dice"], metrics["iou"]))

        with open(CSV_LOG, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, train_loss, metrics["dice"], metrics["iou"], metrics["precision"],
                metrics["recall"], metrics["accuracy"], metrics["specificity"],
                metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"], round(duration, 2)
            ])

    # Eğitim Eğrisi
    epochs = [e[0] for e in epoch_losses]
    dice_scores = [e[2] for e in epoch_losses]
    ious = [e[3] for e in epoch_losses]
    plt.plot(epochs, dice_scores, label="Dice")
    plt.plot(epochs, ious, label="IoU")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Dice & IoU over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_PATH, dpi=200)
    print(f"\n📈 Eğitim eğrisi kaydedildi: {PLOT_PATH}")

    # Ekstra: İlk validation batch için tahmin dağılımı kontrolü
    print("\n🔍 Tahmin Dağılımı (Validation'dan 1 örnek):")
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            pred = torch.sigmoid(model(x))
            print("Max prediction:", pred.max().item())
            print("Mean prediction:", pred.mean().item())
            break

if __name__ == "__main__":
    main()
