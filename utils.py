import os
import numpy as np
import cv2
import torch
import torchvision
from dataset import LungDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from model import TverskyFocalLoss

USE_MORF = False
kernel_size = 3

matplotlib.use('Agg')


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = LungDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = LungDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def check_accuracy(loader, model, device="cuda"):
    model.eval()
    total_tp = torch.tensor(0.0).to(device)
    total_fp = torch.tensor(0.0).to(device)
    total_fn = torch.tensor(0.0).to(device)
    total_tn = torch.tensor(0.0).to(device)
    total_loss = 0.0  # Yeni: Validation loss'u hesapla
    loss_fn = TverskyFocalLoss().to(device)  # Loss fonksiyonu

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            preds = model(x)
            loss = loss_fn(preds, y)  # Loss hesapla
            total_loss += loss.item()

            preds = torch.sigmoid(model(x))
            preds_bin = (preds > 0.5).float()

            # TP, FP, FN, TN hesapla
            tp = (preds_bin * y).sum()
            fp = (preds_bin * (1 - y)).sum()
            fn = ((1 - preds_bin) * y).sum()
            tn = ((1 - preds_bin) * (1 - y)).sum()

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

    # Tüm metrikleri hesapla
    dice = (2 * total_tp) / (total_tp + total_fp + total_tp + total_fn + 1e-8)
    iou = total_tp / (total_tp + total_fp + total_fn + 1e-8)
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + 1e-8)
    specificity = total_tn / (total_tn + total_fp + 1e-8)
    avg_val_loss = total_loss / len(loader)  # Ortalama validation loss

    print(f"Dice: {dice.item():.4f} | IoU: {iou.item():.4f} | Precision: {precision.item():.4f} | "
          f"Recall: {recall.item():.4f} | Accuracy: {accuracy.item():.4f} | Specificity: {specificity.item():.4f}")
    print(f"FP: {total_fp.item()} | FN: {total_fn.item()} | FP/FN Ratio: {(total_fp / total_fn).item():.2f}")

    model.train()
    return {
        "dice": dice.item(),
        "iou": iou.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "accuracy": accuracy.item(),
        "specificity": specificity.item(),
        "fp": total_fp.item(),
        "fn": total_fn.item(),
        "val_loss": avg_val_loss
    }


def get_top_predictions(loader, model, device="cuda", top_n=10):
    """Collect top N predictions based on Dice score"""
    model.eval()
    top_scores = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.unsqueeze(1).to(device)
            preds = torch.sigmoid(model(x))
            preds_bin = (preds > 0.5).float()

            for i in range(x.shape[0]):
                intersection = (preds_bin[i] * y[i]).sum()
                pred_sum = preds_bin[i].sum()
                target_sum = y[i].sum()
                dice = (2 * intersection) / (pred_sum + target_sum + 1e-8).item()

                if len(top_scores) < top_n or dice > top_scores[-1][0]:
                    top_scores.append((
                        dice,
                        x[i].cpu(),
                        preds_bin[i].cpu(),
                        y[i].cpu()
                    ))
                    # Sort and keep only top N
                    top_scores.sort(reverse=True, key=lambda x: x[0])
                    top_scores = top_scores[:top_n]

    model.train()
    return top_scores


def save_top_predictions(top_predictions, folder="saved_images/", alpha=0.3):
    """Save top predictions with better overlay visualization"""
    os.makedirs(folder, exist_ok=True)

    # Clear previous images (keep your original clearing logic)
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    for idx, (dice, image, pred, target) in enumerate(top_predictions):
        # Keep your original tensor conversions
        image_np = image.squeeze().numpy()
        pred_np = pred.squeeze().numpy()
        target_np = target.squeeze().numpy()

        if USE_MORF is True:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            cleaned_pred = cv2.morphologyEx(pred_np, cv2.MORPH_OPEN, kernel)
            pred_np = cleaned_pred

        # 1. Normalize just for visualization (keep original tensors intact)
        vis_image = np.clip(image_np, np.percentile(image_np, 5), np.percentile(image_np, 95))
        vis_image = (vis_image - vis_image.min()) / (vis_image.max() - vis_image.min() + 1e-8)

        # 2. Create RGB version with better contrast
        img_rgb = cv2.cvtColor((vis_image * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

        # 3. Create more vibrant overlays
        overlay = img_rgb.copy()
        overlay[target_np > 0.5] = [50, 255, 50]  # Brighter green
        overlay[pred_np > 0.5] = [255, 50, 50]  # Brighter red

        # 4. Better blending with original
        blended = cv2.addWeighted(img_rgb, 1 - alpha, overlay, alpha, 0)
        # --------------------------------------------------------

        fig, axs = plt.subplots(1, 4, figsize=(20, 5))

        # Only change colormap for better visibility
        axs[0].imshow(image_np, cmap='gray', vmin=np.percentile(image_np, 1), vmax=np.percentile(image_np, 99))
        axs[0].set_title('Input Scan')
        axs[0].axis('off')

        axs[1].imshow(pred_np, cmap='Reds')  # Changed to red colormap
        axs[1].set_title(f'Prediction (Dice: {dice:.2f})')
        axs[1].axis('off')

        axs[2].imshow(target_np, cmap='Greens')  # Changed to green colormap
        axs[2].set_title('Ground Truth')
        axs[2].axis('off')

        axs[3].imshow(blended)
        axs[3].set_title('Overlay (Red=Pred, Green=GT)')
        axs[3].axis('off')

        plt.tight_layout()
        plt.savefig(f"{folder}/top_{idx + 1}.png", bbox_inches='tight', dpi=100)
        plt.close()

        # Keep your original tensor saving
        torchvision.utils.save_image(image, f"{folder}/input_{idx + 1}.png")
        torchvision.utils.save_image(pred, f"{folder}/pred_{idx + 1}.png")
        torchvision.utils.save_image(target, f"{folder}/target_{idx + 1}.png")


def visualize_metrics(csv_path, save_dir="plots"):
    """Tüm metrikleri tek bir figürde alt grafikler olarak görselleştir"""
    try:
        df = pd.read_csv(csv_path)

        # Klasörü oluştur
        os.makedirs(save_dir, exist_ok=True)

        # Grafik ayarları
        plt.style.use('seaborn-v0_8')
        metrics = [
            ('learning_rate', 'Learning Rate', 'darkviolet'),
            ('train_loss', 'Train Loss', 'darkblue'),
            ('val_loss', 'Validation Loss', 'navy'),
            ('val_dice', 'Validation Dice', 'forestgreen'),
            ('val_precision', 'Precision', 'mediumpurple'),
            ('val_recall', 'Recall', 'goldenrod'),
            ('val_accuracy', 'Accuracy', 'firebrick'),
            ('val_specificity', 'Specificity', 'teal'),
            ('val_iou', 'IoU', 'coral'),
            ('val_fp', 'False Positives', 'red'),
            ('val_fn', 'False Negatives', 'navy'),
        ]

        # 4x3 grid oluştur (12 subplot, 10 metrik için)
        fig, axs = plt.subplots(4, 3, figsize=(25, 20))
        axs = axs.ravel()  # Düzleştirilmiş eksen dizisi

        for idx, (col, title, color) in enumerate(metrics):
            ax = axs[idx]
            ax.plot(df['epoch'], df[col], color=color, linewidth=2, marker='o', markersize=4)
            ax.set_title(title, fontsize=14, pad=10)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(title, fontsize=12)
            ax.grid(True, alpha=0.3)

            # Özel ayarlar
            if col == 'learning_rate':
                ax.set_yscale('log')
            if col in ['val_fp', 'val_fn']:
                ax.set_ylim(0, df[[col]].max().values[0] * 1.1)

        # Kalan boş subplot'ları kapat
        for idx in range(len(metrics), len(axs)):
            axs[idx].axis('off')

        plt.tight_layout(pad=3.0)
        plt.savefig(f"{save_dir}/all_metrics_grid.png", dpi=200, bbox_inches='tight')
        plt.close()

        print(f"✓ Tüm metrikler '{save_dir}/all_metrics_grid.png' olarak kaydedildi")

    except Exception as e:
        print(f"✗ Görselleştirme hatası: {str(e)}")
