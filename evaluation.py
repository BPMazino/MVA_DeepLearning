import torch
import torch.nn as nn
from utils import compute_map


def test(model, test_loader, device, iou_threshold=0.5):
    """
    Evaluate the model on the test dataset.

    Parameters:
    - model: Trained YOLO model
    - test_loader: DataLoader for test data
    - device: Device to run evaluation on (CPU or GPU)
    - iou_threshold: IoU threshold for mAP computation

    Returns:
    - mAP: Mean Average Precision on the test set
    """
    model.eval()
    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            predictions = model(images)

            all_predictions.append(predictions.cpu())
            all_ground_truths.append(labels.cpu())

    mAP = compute_map(all_predictions, all_ground_truths, iou_threshold)
    print(f"Test mAP@{iou_threshold}: {mAP:.4f}")
    return mAP
