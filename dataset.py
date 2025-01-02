import os
import xml.etree.ElementTree as ET
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from preprocessing import parse_voc_annotation, preprocess_image, prepare_labels, draw_bounding_boxes
import matplotlib.pyplot as plt
import torchvision.transforms as T
import random 

def read_image_ids(split_file):
    """
    Reads image IDs from a split file.

    Parameters:
    - split_file (str): Path to the split file (e.g., train.txt, val.txt, etc.).

    Returns:
    - List of image IDs.
    """
    if not os.path.exists(split_file):
        print(f"Warning: Split file {split_file} not found.")
        return []
    with open(split_file, "r") as f:
        image_ids = f.read().strip().split("\n")
    return image_ids


def get_predefined_splits(image_sets_dir):
    """
    Loads predefined train, val, and test splits from Pascal VOC.

    Parameters:
    - image_sets_dir (str): Path to the ImageSets/Main/ folder.

    Returns:
    - train_ids, val_ids, test_ids (lists of image IDs for each split).
    """
    train_file = os.path.join(image_sets_dir, "train.txt")
    val_file = os.path.join(image_sets_dir, "val.txt")
    test_file = os.path.join(image_sets_dir, "test.txt")

    train_ids = read_image_ids(train_file)
    val_ids = read_image_ids(val_file)
    test_ids = read_image_ids(test_file) if os.path.exists(test_file) else []

    return train_ids, val_ids, test_ids


class VOCDataset(Dataset):
    """
    Pascal VOC Dataset for YOLO training.

    Parameters:
    - image_dir (str): Path to the image directory.
    - annotation_dir (str): Path to the annotation directory.
    - class_names (list): List of class names.
    - image_ids (list): List of image IDs for this dataset split.
    - transform (callable): Transform to apply to images.

    Returns:
    - image (Tensor): Preprocessed image.
    - labels or boxes (Tensor): YOLO labels or bounding boxes.
    """
    def __init__(self, image_dir, annotation_dir, class_names, image_ids, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.class_names = class_names
        self.image_ids = image_ids  # Use predefined splits
        self.transform = transform

        # Check if directories exist
        if not os.path.exists(image_dir) or not os.path.exists(annotation_dir):
            raise Exception("Image or annotation directory does not exist.")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx, return_boxes=False):
        """
        Fetch an item from the dataset.

        Parameters:
        - idx (int): Index of the data to fetch.
        - return_boxes (bool): If True, return bounding boxes instead of YOLO labels.

        Returns:
        - image (Tensor): Preprocessed image.
        - labels or boxes (Tensor): YOLO labels or bounding boxes.
        """
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        annotation_path = os.path.join(self.annotation_dir, f"{image_id}.xml")

        boxes = parse_voc_annotation(annotation_path, self.class_names)
        image, normalized_boxes = preprocess_image(image_path, boxes)
        
        if self.transform:
            image = self.transform(image)  # Apply transform on PIL image
        
        # Convert to tensor after transformations
        image = T.ToTensor()(image)

        if return_boxes:
            return image, torch.tensor(normalized_boxes, dtype=torch.float32)

        labels = prepare_labels(normalized_boxes)
        return image, labels

class VOCDatasetAugmented(VOCDataset):
    def __init__(self, image_dir, annotation_dir, class_names, transform=None):
        super().__init__(image_dir, annotation_dir, class_names, transform)
        self.color_jitter = T.ColorJitter(brightness=1.5, contrast=1.5, saturation=1.5)
            

    def augment(self, image, boxes):
        """
        Apply random augmentation to the image and bounding boxes.

        Parameters:
        - image (torch.Tensor): Input image tensor.
        - boxes (torch.Tensor): Bounding boxes tensor in normalized format.

        Returns:
        - image (torch.Tensor): Augmented image tensor.
        - boxes (torch.Tensor): Augmented bounding boxes tensor.
        """
        # Apply random HSV adjustments
        image = self.color_jitter(image)

        # Get original dimensions
        height, width = image.shape[-2:]

        # Random horizontal flip
        if random.random() < 0.5:
            image = T.functional.hflip(image)
            boxes[:, 0] = 1 - boxes[:, 0]  # Flip x-center

        # Random scaling
        scale = random.uniform(0.8, 1.2)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = T.functional.resize(image, (new_height, new_width))

        # Scale bounding boxes
        boxes[:, [0, 2]] *= new_width / width
        boxes[:, [1, 3]] *= new_height / height

        # Random translation (proportional to new dimensions)
        dx = random.uniform(-0.2, 0.2) * new_width
        dy = random.uniform(-0.2, 0.2) * new_height
        boxes[:, 0] += dx / new_width
        boxes[:, 1] += dy / new_height

        # Clip boxes to valid range [0, 1]
        boxes[:, :4] = torch.clamp(boxes[:, :4], 0, 1)

        # Final resize to 448x448
        image = T.functional.resize(image, (448, 448))
        scale_x = 448 / new_width
        scale_y = 448 / new_height
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        # Remove invalid boxes (width or height <= 0)
        boxes = boxes[torch.all(boxes[:, 2:] > 0, dim=1)]

        return image, boxes


    
    
    def __getitem__(self, idx):
        image, boxes = super().__getitem__(idx, return_boxes=True)

        image, boxes = self.augment(image, boxes)

        labels = prepare_labels(boxes)
        return image, labels


""" 

# Paths to Pascal VOC dataset
image_dir = "VOC_dataset/VOCdevkit/VOC2007/JPEGImages"  # Replace with your path
annotation_dir = "VOC_dataset/VOCdevkit/VOC2007/Annotations"  # Replace with your path

# Pascal VOC class names
class_names = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", 
    "bus", "car", "cat", "chair", "cow", "diningtable", 
    "dog", "horse", "motorbike", "person", "pottedplant", 
    "sheep", "sofa", "train", "tvmonitor"
]

# Initialize dataset and dataloader
voc_dataset = VOCDatasetAugmented(image_dir, annotation_dir, class_names)
dataloader = DataLoader(voc_dataset, batch_size=1, shuffle=True)


for i, (image, labels) in enumerate(dataloader):
    print(f"Image shape: {image.shape}")  # Expect: (batch_size, 3, 448, 448)
    print(f"Labels shape: {labels.shape}")  # Expect: (batch_size, 7, 7, 25)


    # Visualize the image (convert tensor to NumPy for visualization)
    image_np = image[0].permute(1, 2, 0).numpy()
    plt.imshow(image_np)
    plt.title("Sample Image")
    plt.axis("off")
    plt.show()

    # Debugging labels
    print("Labels (Grid Cell [x, y, w, h, conf, ...]):")
    print(labels[0, :3, :3, :])  # Print a subset of the labels for debugging

    if i == 0:
        break
    
draw_bounding_boxes(image[0], labels[0]) """

