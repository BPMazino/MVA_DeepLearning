# YOLO utils functions

import cv2
import torch

def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def resize_image(image, size = (448, 448)):
    return cv2.resize(image, size)


def compute_iou(box1,box2): 
    """ 
    Compute the Intersection over Union (IoU) of two bounding boxes
    
    Parameters:
    box1: list of 4 elements [x1, y1, x2, y2]
    box2: list of 4 elements [x1, y1, x2, y2]
    
    Returns:
    iou: float, Intersection over Union (IoU) of box1 and box2
    """
    

    x1_min, y1_min = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    x1_max, y1_max = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    x2_min, y2_min = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    x2_max, y2_max = box2[0] + box2[2]/2, box2[1] + box2[3]/2
    
    xA = max(x1_min, x2_min)
    yA = max(y1_min, y2_min)
    xB = min(x1_max, x2_max)
    yB = min(y1_max, y2_max)
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    
    iou = interArea / (area1 + area2 - interArea)
    
    return max(0, iou)


def non_max_suppression(boxes, confidences, iou_threshold):
    """ 
    Apply Non-maximum Suppression (NMS) to bounding boxes
    
    Parameters:
    boxes: list of bounding boxes, each box is a list of 4 elements [x, y, w, h]
    confidences: list of confidences for each box
    iou_threshold: float, Intersection over Union (IoU) threshold for suppressing overlapping boxes
    
    Returns:
    boxes: list of remaining after NMS
    """
    if len(boxes) == 0:
        return []
    sorted_indices = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)
    boxes = [boxes[i] for i in sorted_indices]
    
    retained_boxes = []
    
    while boxes:
        max_confidence_box = boxes.pop(0)
        retained_boxes.append(max_confidence_box)
        boxes = [box for box in boxes if compute_iou(max_confidence_box, box) < iou_threshold]
    
    return retained_boxes


def compute_map(predictions,ground_truths, iou_threshold = 0.5):
    """
    Compute the Mean Average Precision (mAP) of a model

    Parameters:
    predictions: list of predicted bounding boxes for each image
    ground_truths: list of ground truth bounding boxes for each image
    iou_threshold: float, Intersection over Union (IoU) threshold for matching predictions to ground truths
    """
    
    aps = []
    
    for class_id in range(len(predictions)):
        preds = predictions[class_id]
        gts = ground_truths[class_id]
        
        preds = sorted(preds, key=lambda x: x[4], reverse=True)
        
        tp = [0] * len(preds)
        fp = [0] * len(preds)
        
        for i, pred in enumerate(preds):
            best_iou = 0
            best_gt = None
            for gt in gts:
                iou = compute_iou(pred[:4], gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt
            
            if best_iou > iou_threshold:
                tp[i] = 1
                gts.remove(best_gt)
            else:
                fp[i] = 1
                
        tp_cumsum = torch.cumsum(torch.tensor(tp), dim=0)
        fp_cumsum = torch.cumsum(torch.tensor(fp), dim=0)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        recalls = tp_cumsum / len(gts) + 1e-6
        
        ap  = torch.trapz(precisions, recalls)
        aps.append(ap)

    return torch.mean(torch.tensor(aps))