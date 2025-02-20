"""
Implementation of the YOLO Loss Function from the original YOLO paper.
"""

import torch
import torch.nn as nn
from typing import Tuple
from utils import intersection_over_union


class YoloLoss(nn.Module):
    """
    Computes the YOLOv1 loss, which is composed of:
        - Coordinate loss for bounding box predictions
        - Object loss for cells containing objects
        - No-object loss for cells not containing objects
        - Classification loss for the predicted classes
    """
    def __init__(self, S: int = 7, B: int = 2, C: int = 20) -> None:
        """
        Args:
            S (int): The grid size (default: 7).
            B (int): The number of bounding boxes per grid cell (default: 2).
            C (int): The number of classes (default: 20).
        """
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        self.S = S
        self.B = B
        self.C = C

        # Loss scaling factors from the YOLO paper:
        self.lambda_noobj = 0.5  # scales the no-object confidence loss
        self.lambda_coord = 5    # scales the bounding box coordinate loss

    def forward(self, predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the YOLO loss.

        Args:
            predictions (torch.Tensor): Predictions of shape (batch_size, S*S*(C+B*5)).
            target (torch.Tensor): Ground truth tensor with the same shape as predictions.

        Returns:
            torch.Tensor: The computed total loss.
        """
        # Reshape predictions to (batch_size, S, S, C + B * 5)
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # ---------------------------------------------------------------------
        # 1. Determine the best bounding box for each cell based on IoU.
        # ---------------------------------------------------------------------
        # For the first bounding box: indices 21:25 are [x, y, w, h] and index 20 is the confidence.
        # For the second bounding box: indices 26:30 are [x, y, w, h] and index 25 is the confidence.
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        # Stack IoUs to compare which predicted box is better per grid cell.
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)  # Shape: (2, batch, S, S)

        # Determine the index (0 or 1) of the bounding box with the highest IoU.
        _, bestbox = torch.max(ious, dim=0)  # Shape: (batch, S, S)
        # Create an indicator (Iobj) for cells that contain an object. Shape: (batch, S, S, 1)
        exists_box = target[..., 20].unsqueeze(3)

        # ---------------------------------------------------------------------
        # 2. Compute the Coordinate (Box) Loss
        # ---------------------------------------------------------------------
        # Use bestbox to select the appropriate predicted bounding box.
        # If bestbox == 0, select the first bounding box; if bestbox == 1, select the second.
        bestbox = bestbox.unsqueeze(-1)  # Adjust shape for broadcasting: (batch, S, S, 1)
        pred_box_coords = exists_box * (
            bestbox * predictions[..., 26:30] + (1 - bestbox) * predictions[..., 21:25]
        )
        target_box_coords = exists_box * target[..., 21:25]

        # Apply square root to the width and height to reduce the impact of large errors.
        epsilon = 1e-6  # small constant to avoid numerical instability
        pred_box_coords[..., 2:4] = torch.sign(pred_box_coords[..., 2:4]) * torch.sqrt(
            torch.abs(pred_box_coords[..., 2:4] + epsilon)
        )
        target_box_coords[..., 2:4] = torch.sqrt(target_box_coords[..., 2:4])

        box_loss = self.mse(
            torch.flatten(pred_box_coords, end_dim=-2),
            torch.flatten(target_box_coords, end_dim=-2)
        )

        # ---------------------------------------------------------------------
        # 3. Compute the Object Loss (Confidence for cells with objects)
        # ---------------------------------------------------------------------
        # Select the confidence score from the best bounding box.
        pred_box_conf = bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box_conf),
            torch.flatten(exists_box * target[..., 20:21])
        )

        # ---------------------------------------------------------------------
        # 4. Compute the No-Object Loss (Confidence for cells without objects)
        # ---------------------------------------------------------------------
        # For cells that do not contain objects, the target confidence is 0.
        noobj_pred_conf_first = predictions[..., 20:21]
        noobj_pred_conf_second = predictions[..., 25:26]
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * noobj_pred_conf_first, start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )
        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * noobj_pred_conf_second, start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        # ---------------------------------------------------------------------
        # 5. Compute the Classification Loss
        # ---------------------------------------------------------------------
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :self.C], end_dim=-2),
            torch.flatten(exists_box * target[..., :self.C], end_dim=-2)
        )

        # ---------------------------------------------------------------------
        # Total Loss: Weighted sum of all the individual losses.
        # ---------------------------------------------------------------------
        total_loss = (
            self.lambda_coord * box_loss +
            object_loss +
            self.lambda_noobj * no_object_loss +
            class_loss
        )

        return total_loss
