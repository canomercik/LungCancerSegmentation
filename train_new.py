import os
import shutil
import csv
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET, TverskyFocalLoss, DiceFocalLoss
from utils import (load_checkpoint, get_loaders, check_accuracy,
                   save_top_predictions, get_top_predictions, visualize_metrics)
import time
import json
#import torch.nn.functional as F
#from lovasz_losses import lovasz_softmax


# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 50
NUM_WORKERS = 4
IMAGE_HEIGHT = 128  # 512 originally
IMAGE_WIDTH = 128  # 512 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "train/images/"
TRAIN_MASK_DIR = "train/masks/"
VAL_IMG_DIR = "val/images/"
VAL_MASK_DIR = "val/masks/"

# Gates
USE_AUG = True

# Early stopping parameters
patience = 6  # Number of epochs to wait for improvement before stopping early
patience_counter = 0


def create_run_folders(base_dir="run"):
    """
    Creates a new run folder (e.g., run/train1, run/train2, ...) with subfolders:
      - weights: to store best and last model checkpoints
      - test_photos: to store test images
    Returns:
      run_folder, weights_folder, test_folder
    """
    run_id = 1
    while os.path.exists(os.path.join(base_dir, f"train{run_id}")):
        run_id += 1
    run_folder = os.path.join(base_dir, f"train{run_id}")
    os.makedirs(run_folder)

    weights_folder = os.path.join(run_folder, "weights")
    os.makedirs(weights_folder)

    test_folder = os.path.join(run_folder, "test_photos")
    os.makedirs(test_folder)

    return run_folder, weights_folder, test_folder


def save_checkpoint(state, weights_folder, is_best, best_filename="best_model.pth.tar",
                    last_filename="last_model.pth.tar"):
    last_path = os.path.join(weights_folder, last_filename)
    torch.save(state, last_path)
    if is_best:
        best_path = os.path.join(weights_folder, best_filename)
        shutil.copyfile(last_path, best_path)
        print("=> Saved best model checkpoint at", best_path)


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    total_loss = 0.0  # Toplam kaybı takip etmek için

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # Forward pass
        with torch.amp.autocast('cuda'):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Kaybı güncelle
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item(), batch=batch_idx)

    # Ortalama kaybı hesapla
    avg_train_loss = total_loss / len(loader)
    return avg_train_loss  # Ortalama kaybı döndür


def main():
    global patience_counter  # to update the counter in our early stopping mechanism

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")

    # Create new run folders for weights and test images
    run_folder, weights_folder, test_folder = create_run_folders()
    print("Run folder created:", run_folder)

    # Prepare CSV results file in the run folder
    csv_path = os.path.join(run_folder, "results.csv")
    with open(csv_path, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write header
        csv_writer.writerow([
            "epoch",
            "train_loss",
            "val_dice",
            "val_iou",
            "val_precision",
            "val_recall",
            "val_accuracy",
            "val_specificity",
            "val_fp",
            "val_fn",
            "val_loss",
            "best_val_dice",
            "learning_rate",
            "epoch_time"
        ])
    if USE_AUG is True:
        train_transform = A.Compose([
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=15, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomBrightnessContrast(p=0.2),  # Yeni
            A.GaussNoise(var=(0.001, 0.005), p=0.3),  # Yeni
            A.ElasticTransform(alpha=1, sigma=50, affine=None, p=0.3),  # Lezyon çeşitliliği AZALTILABİLİR.
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.Normalize(mean=(0.0), std=(1.0), max_pixel_value=255.0),
            ToTensorV2(),
        ])
    else:
        train_transform = A.Compose(
            [
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Rotate(limit=15, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0),  # Tek kanal görüntü için
                ToTensorV2(),
            ],
        )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=1, out_channels=1).to(DEVICE)
    # loss_fn = DiceFocalLoss(alpha=0.8, gamma=2.0).to(DEVICE)
    loss_fn = TverskyFocalLoss(alpha=0.3, beta=0.7).to(DEVICE)
    # loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Learning Rate Scheduler ekleyin (ReduceLROnPlateau örneği)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # Val dice'ı takip ediyoruz (maximizasyon)
        factor=0.1,  # Öğrenme oranını 10 kat azalt
        patience=3,  # 3 epoch iyileşme olmazsa tetikle
        verbose=True  # Bilgilendirme mesajı göster
    )

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    # Initial evaluation before training
    _ = check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.amp.GradScaler('cuda')  # Corrected initialization

    best_val_dice = 0.0  # Track the best validation Dice score

    # Save training arguments for future reference/resume
    training_args = {
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "image_height": IMAGE_HEIGHT,
        "image_width": IMAGE_WIDTH,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
        "train_img_dir": TRAIN_IMG_DIR,
        "train_mask_dir": TRAIN_MASK_DIR,
        "val_img_dir": VAL_IMG_DIR,
        "val_mask_dir": VAL_MASK_DIR,
    }

    with open(os.path.join(run_folder, "training_args.json"), "w") as f:
        json.dump(training_args, f, indent=4)

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()  # Epoch başlangıç zamanı

        print("Epoch ==> ", epoch + 1, "/", NUM_EPOCHS)
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # Evaluate on validation set and obtain validation Dice score.
        # Assumption: check_accuracy returns the dice score.
        metrics = check_accuracy(val_loader, model, device=DEVICE)
        scheduler.step(metrics["dice"])

        # Epoch süresi
        epoch_time = time.time() - start_time

        # Prepare the checkpoint state with extra training metadata
        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_dice": best_val_dice,
            "training_args": training_args,
        }

        # Determine if the current model is the best so far
        is_best = metrics["dice"] > best_val_dice
        if is_best:
            best_val_dice = metrics["dice"]
            patience_counter = 0  # Reset patience counter if improvement
        else:
            patience_counter += 1

        # Save the checkpoint in the weights folder
        save_checkpoint(checkpoint, weights_folder, is_best)

        # If it's the best model, save top test predictions in the test photos folder
        if is_best:
            print("Saving test predictions for best model...")
            top_predictions = get_top_predictions(val_loader, model, device=DEVICE, top_n=5)
            save_top_predictions(top_predictions, folder=test_folder)

        # Log metrics to CSV file
        with open(csv_path, mode="a", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                epoch + 1,
                train_loss,  # Ortalama eğitim kaybı
                metrics["dice"],
                metrics["iou"],
                metrics["precision"],
                metrics["recall"],
                metrics["accuracy"],
                metrics["specificity"],
                metrics["fp"],
                metrics["fn"],
                metrics["val_loss"],
                best_val_dice,
                optimizer.param_groups[0]["lr"],  # Mevcut öğrenme oranı
                epoch_time
            ])
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}: ")
        print("Dice Score : " + str(metrics["dice"]))
        # Early stopping check
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

    print("Training complete. Results saved in", csv_path)
    csv_path = os.path.join(run_folder, "results.csv")
    visualize_metrics(csv_path, save_dir=os.path.join(run_folder, "metrics_plots"))


if __name__ == "__main__":
    main()
