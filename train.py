import os
from typing import Any, List, Tuple

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm

from dataset import VOCDataset
from model import Yolov1
from loss import YoloLoss
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)

# --------------------------- Hyperparameters --------------------------- #
SEED: int = 123
torch.manual_seed(SEED)

LEARNING_RATE: float = 2e-5
WEIGHT_DECAY: float = 0
BATCH_SIZE: int = 16
NUM_EPOCHS: int = 100
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS: int = 2
LOAD_MODEL: bool = False
CHECKPOINT_FILE: str = "checkpoint.pth.tar"

# Paths for images, labels, and annotation CSVs
IMG_DIR: str = "data/images"
LABEL_DIR: str = "data/labels"
TRAIN_CSV: str = "data/train.csv"
TEST_CSV: str = "data/test.csv"

# YOLO v1 specific parameters
S: int = 7  # Grid size
B: int = 2  # Number of bounding boxes per cell
C: int = 20  # Number of classes (Pascal VOC)


# -------------------------- Compose Transforms -------------------------- #
class Compose:
    """
    Composes several transforms sequentially.

    This class applies a list of transforms to the image while leaving the bounding
    boxes unchanged (but can be extended to modify boxes as well).
    """
    def __init__(self, transforms_list: List[Any]) -> None:
        self.transforms_list = transforms_list

    def __call__(self, img: Any, bboxes: torch.Tensor) -> Tuple[Any, torch.Tensor]:
        for transform in self.transforms_list:
            img = transform(img)
        return img, bboxes


# Define basic transformations: Resize images to 448x448 and convert them to tensors.
train_transforms = Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])


# ----------------------------- Train Function ----------------------------- #
def train_fn(
    train_loader: DataLoader, 
    model: nn.Module, 
    optimizer: optim.Optimizer, 
    loss_fn: YoloLoss
) -> float:
    """
    Executes one training epoch.

    Parameters:
        train_loader (DataLoader): Provides batches of (image, label_matrix) pairs.
        model (nn.Module): The YOLOv1 model.
        optimizer (optim.Optimizer): Optimizer for model parameters.
        loss_fn (YoloLoss): The YOLO loss function.

    Returns:
        float: The average loss for the epoch.
    """
    model.train()  # Set model to training mode
    epoch_losses: List[float] = []
    progress_bar = tqdm(train_loader, leave=True)

    for images, labels in progress_bar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # Forward pass
        predictions = model(images)
        loss = loss_fn(predictions, labels)
        epoch_losses.append(loss.item())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update progress bar
        progress_bar.set_postfix(loss=loss.item())

    return sum(epoch_losses) / len(epoch_losses)


# ------------------------------- Main Loop ------------------------------- #
def main() -> None:
    """
    Main training script for YOLOv1 on Pascal VOC-style data.

    Workflow:
        1. Initialize model, optimizer, and loss function.
        2. Optionally load a checkpoint.
        3. Create training and test DataLoaders.
        4. For each epoch:
            - Compute mAP on the training set.
            - Run a training epoch.
            - Save checkpoints periodically.
        5. Perform a final evaluation on the test set.
    """
    # --------- Initialize Model, Optimizer, and Loss --------- #
    model = Yolov1(split_size=S, num_boxes=B, num_classes=C).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        print("=> Loading checkpoint")
        load_checkpoint(torch.load(CHECKPOINT_FILE), model, optimizer)

    # --------------- Create Datasets & DataLoaders --------------- #
    train_dataset = VOCDataset(
        csv_file=TRAIN_CSV,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        S=S, B=B, C=C,
        transform=train_transforms
    )
    test_dataset = VOCDataset(
        csv_file=TEST_CSV,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        S=S, B=B, C=C,
        transform=train_transforms
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        shuffle=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    # --------------- Training Loop --------------- #
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        # Compute mAP on the training set (optional, can be time-consuming)
        print("Generating predictions for mAP calculation on training set...")
        pred_boxes, target_boxes = get_bboxes(
            train_loader,
            model,
            iou_threshold=0.5,
            threshold=0.4,     # Confidence threshold
            device=DEVICE,
        )

        print("Computing mAP for training set...")
        train_map = mean_average_precision(
            pred_boxes,
            target_boxes,
            iou_threshold=0.5,
            box_format="midpoint",
            num_classes=C,
        )
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] -- Train mAP: {train_map:.4f}")

        # Execute one training epoch
        avg_loss = train_fn(train_loader, model, optimizer, loss_fn)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] -- Mean Loss: {avg_loss:.4f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=CHECKPOINT_FILE)
            print("=> Saved checkpoint")

    # --------------- Final Evaluation on Test Set --------------- #
    model.eval()
    pred_boxes, target_boxes = get_bboxes(
        test_loader,
        model,
        iou_threshold=0.5,
        threshold=0.4,
        device=DEVICE,
    )
    test_map = mean_average_precision(
        pred_boxes,
        target_boxes,
        iou_threshold=0.5,
        box_format="midpoint",
        num_classes=C,
    )
    print(f"Final Test mAP: {test_map:.4f}")


if __name__ == "__main__":
    main()
