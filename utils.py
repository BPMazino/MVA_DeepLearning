import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
from typing import List, Tuple, Any

def intersection_over_union(
    boxes_preds: torch.Tensor,
    boxes_labels: torch.Tensor,
    box_format: str = "midpoint"
) -> torch.Tensor:
    """
    Calculates Intersection over Union (IoU) between predicted and target bounding boxes.

    Parameters:
        boxes_preds (torch.Tensor): Predicted bounding boxes of shape (..., 4).
        boxes_labels (torch.Tensor): Ground truth bounding boxes of shape (..., 4).
        box_format (str): Format of bounding boxes. Use "midpoint" for (x, y, w, h) 
                          or "corners" for (x1, y1, x2, y2).

    Returns:
        torch.Tensor: IoU for each bounding box.
    """
    epsilon = 1e-6

    if box_format == "midpoint":
        # Convert (x, y, w, h) to (x1, y1, x2, y2)
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]

        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]
    else:
        raise ValueError(f"Invalid box_format: {box_format}. Expected 'midpoint' or 'corners'.")

    # Calculate coordinates of the intersection rectangle
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Clamp values for non-overlapping boxes and compute intersection area
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = torch.abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = torch.abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + epsilon)


def non_max_suppression(
    bboxes: List[List[float]],
    iou_threshold: float,
    threshold: float,
    box_format: str = "corners"
) -> List[List[float]]:
    """
    Performs Non-Maximum Suppression (NMS) on a list of bounding boxes.

    Parameters:
        bboxes (List[List[float]]): List of bounding boxes, each specified as 
            [class_pred, prob_score, x1, y1, x2, y2].
        iou_threshold (float): IoU threshold for NMS.
        threshold (float): Confidence threshold to filter bounding boxes.
        box_format (str): Format of bounding boxes - "midpoint" or "corners".

    Returns:
        List[List[float]]: Bounding boxes after NMS.
    """
    if not isinstance(bboxes, list):
        raise ValueError("bboxes should be a list of bounding boxes.")

    # Filter boxes with low confidence
    bboxes = [box for box in bboxes if box[1] > threshold]
    # Sort boxes by confidence score in descending order
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box for box in bboxes
            if box[0] != chosen_box[0] or
               intersection_over_union(
                   torch.tensor(chosen_box[2:]),
                   torch.tensor(box[2:]),
                   box_format=box_format,
               ) < iou_threshold
        ]
        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mean_average_precision(
    pred_boxes: List[List[float]],
    true_boxes: List[List[float]],
    iou_threshold: float = 0.5,
    box_format: str = "midpoint",
    num_classes: int = 20
) -> float:
    """
    Calculates Mean Average Precision (mAP) for object detection.

    Parameters:
        pred_boxes (List[List[float]]): List of predicted bounding boxes, each specified as 
            [train_idx, class_prediction, prob_score, x1, y1, x2, y2].
        true_boxes (List[List[float]]): List of ground truth bounding boxes in the same format.
        iou_threshold (float): IoU threshold to consider a prediction correct.
        box_format (str): Format of bounding boxes - "midpoint" or "corners".
        num_classes (int): Number of classes.

    Returns:
        float: The mean average precision across all classes.
    """
    average_precisions = []
    epsilon = 1e-6  # For numerical stability

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Filter detections and ground truths for the current class
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)
        for gt in true_boxes:
            if gt[1] == c:
                ground_truths.append(gt)

        # Count the number of ground truth boxes per image
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # Sort detections by confidence score in descending order
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_bboxes = len(ground_truths)

        # Skip class if no ground truths exist
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Filter ground truths for the current image
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
            best_iou = 0.0
            best_gt_idx = -1

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # Ensure each ground truth is detected only once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        # Prepend sentinel values for integration
        precisions = torch.cat((torch.tensor([1.0]), precisions))
        recalls = torch.cat((torch.tensor([0.0]), recalls))
        # Compute Average Precision (AP) using numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return float(sum(average_precisions) / (len(average_precisions) + epsilon))


def plot_image(image: np.ndarray, boxes: List[List[float]]) -> None:
    """
    Plots predicted bounding boxes on the given image.

    Parameters:
        image (np.ndarray): The image as a NumPy array.
        boxes (List[List[float]]): List of bounding boxes, each with format 
            [class_pred, confidence, x, y, w, h] where (x, y) is the center.
    """
    im = np.array(image)
    height, width, _ = im.shape

    fig, ax = plt.subplots(1)
    ax.imshow(im)

    for box in boxes:
        # Extract bounding box coordinates (ignoring class and confidence)
        box_coords = box[2:]
        if len(box_coords) != 4:
            raise ValueError("Each bounding box must have 4 values: x, y, w, h.")
        upper_left_x = box_coords[0] - box_coords[2] / 2
        upper_left_y = box_coords[1] - box_coords[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box_coords[2] * width,
            box_coords[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)

    plt.show()


def get_bboxes(
    loader: Any,
    model: torch.nn.Module,
    iou_threshold: float,
    threshold: float,
    pred_format: str = "cells",
    box_format: str = "midpoint",
    device: str = "cuda"
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Retrieves predicted and ground truth bounding boxes from a data loader.

    Parameters:
        loader: DataLoader yielding batches of images and labels.
        model (torch.nn.Module): YOLO model.
        iou_threshold (float): IoU threshold for non-max suppression.
        threshold (float): Confidence threshold for filtering predictions.
        pred_format (str): Format of predictions (default: "cells").
        box_format (str): Format of bounding boxes - "midpoint" or "corners".
        device (str): Device for computation.

    Returns:
        Tuple[List[List[float]], List[List[float]]]: 
            - All predicted bounding boxes.
            - All ground truth bounding boxes.
    """
    all_pred_boxes: List[List[float]] = []
    all_true_boxes: List[List[float]] = []

    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


def convert_cellboxes(predictions: torch.Tensor, S: int = 7) -> torch.Tensor:
    """
    Converts YOLO cell-based predictions to bounding boxes relative to the entire image.

    Parameters:
        predictions (torch.Tensor): Predictions of shape (batch_size, 7*7*30).
        S (int): Grid size (default: 7).

    Returns:
        torch.Tensor: Converted predictions with shape (batch_size, 7, 7, 30), where each box is 
                      represented as [class, confidence, x, y, w, h].
    """
    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)

    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)),
        dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2

    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_h = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_h), dim=-1)

    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(-1)
    converted_preds = torch.cat((predicted_class, best_confidence, converted_bboxes), dim=-1)

    return converted_preds


def cellboxes_to_boxes(out: torch.Tensor, S: int = 7) -> List[List[float]]:
    """
    Converts cell-based bounding box predictions to a list of bounding boxes per image.

    Parameters:
        out (torch.Tensor): Model predictions of shape (batch_size, 7*7*30).
        S (int): Grid size (default: 7).

    Returns:
        List[List[float]]: List of bounding boxes for each image, with each box represented as 
                           [class, confidence, x, y, w, h].
    """
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes: List[List[float]] = []

    for ex_idx in range(out.shape[0]):
        bboxes = []
        for bbox_idx in range(S * S):
            bbox = [x.item() for x in converted_pred[ex_idx, bbox_idx, :]]
            bboxes.append(bbox)
        all_bboxes.append(bboxes)

    return all_bboxes


def save_checkpoint(state: dict, filename: str = "my_checkpoint.pth.tar") -> None:
    """
    Saves the training state to a checkpoint file.

    Parameters:
        state (dict): Training state to be saved.
        filename (str): Filename for the checkpoint.
    """
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(
    checkpoint: dict,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> None:
    """
    Loads the model and optimizer state from a checkpoint.

    Parameters:
        checkpoint (dict): Checkpoint dictionary containing model and optimizer states.
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
    """
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
