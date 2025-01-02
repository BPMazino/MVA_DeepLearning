import cv2
import torch
import numpy as np
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def parse_voc_annotation(ann_path, class_names):
    """
    Parse a single XML file from the PASCAL VOC dataset and return the bounding boxes

    Parameters:
        ann_path (str): path to the annotation file
        class_names (list): names of the classes
        
    Returns:
        boxes (list): list of bounding boxes for the image as [x_min, y_min, x_max, y_max, class_id]   
    """
    
    tree = ET.parse(ann_path)
    root = tree.getroot()
    
    boxes = []
    for obj in root.iter('object'):
        class_name = obj.find('name').text
        if class_name not in class_names:
            continue
        class_id = class_names.index(class_name)
        xml_box = obj.find('bndbox')
        x_min = int(xml_box.find('xmin').text)
        y_min = int(xml_box.find('ymin').text)
        x_max = int(xml_box.find('xmax').text)
        y_max = int(xml_box.find('ymax').text)
        boxes.append([x_min, y_min, x_max, y_max, class_id])
    return boxes
        
    

def preprocess_image(img_path, boxes, image_size = (448, 448)):
    """
    Preprocess an image before feeding into the YOLO model
    
    Parameters:
    img_path: str, path to the image
    boxes: list of bounding boxes, each box is a list [x_min, y_min, x_max, y_max, class_id]
    image_size: tuple, size of the output image (width, height)
    
    Returns:
    img: preprocessed image as a numpy array
    target: normalized coordinates of the bounding boxes [x_center, y_center, width, height, class_id]
    """
    
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_width, original_height = image.shape[1], image.shape[0]
    image = cv2.resize(image, image_size)
    
    # Convert to PIL Image
    image = Image.fromarray(image)
    
    normalized_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max, class_id = box
        x_center = (x_min + x_max) / 2 / original_width
        y_center = (y_min + y_max) / 2 / original_height
        width = (x_max - x_min) / original_width
        height = (y_max - y_min) / original_height
        x_center = max(min(x_center, 1.0), 0.0)
        y_center = max(min(y_center, 1.0), 0.0)
        width = max(min(width, 1.0), 0.0)
        height = max(min(height, 1.0), 0.0)
        normalized_boxes.append([x_center, y_center, width, height, class_id])
    return image, torch.tensor(normalized_boxes, dtype=torch.float32)
        
def prepare_labels(boxes, grid_size = 7, num_classes = 20):
    """
    Prepare the YOLO labels for a single image
    
    Parameters:
    boxes: list of bounding boxes for a single image as [x_center, y_center, width, height, class_id]
    grid_size: int, number of grid cells in each dimension (S)
    num_classes: int, number of classes (C)
    
    Returns:
    labels: YOLO labels for the image as a tensor of shape (grid_size, grid_size, 5 x B + C)
    
    """  

    labels = torch.zeros((grid_size, grid_size, 5 + num_classes), dtype=torch.float32)
    for box in boxes:
        x_center, y_center, width, height, class_id = box
        
        grid_x = int(grid_size * x_center)
        grid_y = int(grid_size * y_center)
        
        grid_x = min(grid_size - 1, int(grid_size * x_center))
        grid_y = min(grid_size - 1, int(grid_size * y_center))

        
        
        x_offset = grid_size * x_center - grid_x
        y_offset = grid_size * y_center  - grid_y
        
        labels[grid_y, grid_x, 0:4] = torch.tensor([x_offset, y_offset, width, height]) # Bounding box coordinates
        labels[grid_y, grid_x, 4] = 1 # Confidence score
        labels[grid_y, grid_x, 5 + int(class_id)] = 1 # one-hot encoding for class  
        
    return labels 


def draw_bounding_boxes(image, labels, grid_size=7, num_classes=20, save=False):
    """
    Draw bounding boxes on the image for visualization.

    Parameters:
    - image: PyTorch tensor of shape (3, H, W).
    - labels: PyTorch tensor of shape (S, S, 5 + num_classes).
    - grid_size: int, number of grid cells (S).
    - num_classes: int, number of classes (C).
    """

    image_np = image.permute(1, 2, 0).numpy()  # Convert to HWC format
    fig, ax = plt.subplots(1)
    ax.imshow(image_np)

    cell_size = 1 / grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            if labels[i, j, 4] > 0:  # Confidence > 0
                x_offset, y_offset, width, height = labels[i, j, :4]
                x_center = (j + x_offset) * cell_size
                y_center = (i + y_offset) * cell_size
                

                w = width * image.shape[2]  # Convert relative width to pixels
                h = height * image.shape[1]  # Convert relative height to pixels

                x_min = (x_center - width / 2) * image.shape[2]
                y_min = (y_center - height / 2) * image.shape[1]

                rect = patches.Rectangle((x_min, y_min), w, h, linewidth=2, edgecolor="red", facecolor="none")
                ax.add_patch(rect)

    plt.show()
    # save the image
    if save:
        plt.savefig('output.png')


