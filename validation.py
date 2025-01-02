import torch
import torch.nn as nn
from preprocessing import draw_bounding_boxes
from utils import non_max_suppression

def visualize_validation(model, val_loader, class_names, device, nms_threshold=0.5):
    model.eval()
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            predictions = model(images)

            for img_idx in range(images.shape[0]):
                boxes = predictions[img_idx, :, :4]
                confidences = predictions[img_idx, :, 4]
                filtered_boxes = non_max_suppression(boxes, confidences, nms_threshold)

                draw_bounding_boxes(images[img_idx].cpu(), filtered_boxes, class_names)
