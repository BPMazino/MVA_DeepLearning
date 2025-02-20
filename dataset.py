import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class VOCDataset(Dataset):
    """
    A PyTorch Dataset for loading Pascal VOC images and labels
    (or similarly formatted data) for YOLO v1.

    Parameters
    ----------
    csv_file : str
        Path to a CSV file with two columns:
          - Column 0: image filename
          - Column 1: corresponding label (text file) filename

    img_dir : str
        Path to the directory containing images.

    label_dir : str
        Path to the directory containing the label text files.

    S : int
        The grid size. YOLO v1 default is 7.

    B : int
        The number of bounding boxes per cell. YOLO v1 default is 2.

    C : int
        Number of classes. Pascal VOC has 20.

    transform : callable, optional
        A function/transform that takes in a PIL image and a list of bboxes
        and returns (transformed_image, transformed_bboxes).
        bboxes are expected to have shape (N, 5) = [class, x, y, w, h].
        x, y, w, h are all float in [0..1]; class is an int.

    Notes
    -----
    - Each label text file has lines of the form:
        class_label x_center y_center width height
      all normalized to [0..1].
    - The output label tensor has shape (S, S, C + 5*B).
      Indexes:
         0..C-1      -> one-hot class probabilities
         C           -> objectness score for B=1 (or first bbox if B=2)
         C+1..C+4    -> bbox coordinates (x, y, w, h)
         C+5         -> objectness score for second bbox (if B=2)
         C+6..C+9    -> second bbox coordinates (x, y, w, h)
      For YOLOv1 with B=2, C=20 => (S, S, 30).
    """

    def __init__(self, csv_file, img_dir, label_dir,
                 S=7, B=2, C=20, transform=None):
        super().__init__()
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.S = S
        self.B = B
        self.C = C
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        """
        Returns
        -------
        image : torch.Tensor
            The transformed image tensor (e.g. shape [3, 448, 448]).

        label_matrix : torch.Tensor
            A tensor of shape (S, S, C + 5*B) containing the target
            for YOLOv1 training.
        """
        # ----------------- Read the label file -----------------
        label_filename = self.annotations.iloc[index, 1]
        label_path = os.path.join(self.label_dir, label_filename)

        boxes = []
        with open(label_path, "r") as f:
            for line in f.readlines():
                class_label, x, y, w, h = line.strip().split()
                class_label = int(float(class_label))  # ensure class is int
                x = float(x)
                y = float(y)
                w = float(w)
                h = float(h)
                boxes.append([class_label, x, y, w, h])

        # ----------------- Read the image file -----------------
        img_filename = self.annotations.iloc[index, 0]
        img_path = os.path.join(self.img_dir, img_filename)
        image = Image.open(img_path).convert("RGB")  # ensure 3 channels

        # Convert to torch tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)  # shape (N, 5)

        # ----------------- Optional Transform -----------------
        if self.transform:
            image, boxes = self.transform(image, boxes)
            # transform should output image as a Torch tensor
            # and boxes in the format [class, x, y, w, h] normalized to [0..1]

        # ----------------- Build Label Matrix -----------------
        # shape => (S, S, C + 5*B). For B=2, C=20 => (S, S, 30).
        label_matrix = torch.zeros((self.S, self.S, self.C + 5*self.B))

        # Populate label_matrix by placing each box into the corresponding cell
        for box in boxes:
            class_label = int(box[0])
            x, y, w, h = box[1:].tolist()

            # Compute which cell the center falls into
            cell_row = int(self.S * y)  # i
            cell_col = int(self.S * x)  # j

            # x_cell, y_cell are the offsets in [0..1]
            # relative to the cell boundaries
            x_cell = self.S * x - cell_col
            y_cell = self.S * y - cell_row

            # w_cell, h_cell = width, height in terms of cell size
            w_cell = w * self.S
            h_cell = h * self.S

            # If that cell is empty (objectness=0), fill it
            if label_matrix[cell_row, cell_col, self.C] == 0:
                # Mark that this cell has an object
                label_matrix[cell_row, cell_col, self.C] = 1.0

                # Fill in bbox coords
                label_matrix[cell_row, cell_col, self.C+1 : self.C+5] = torch.tensor(
                    [x_cell, y_cell, w_cell, h_cell]
                )

                # Set the class one-hot
                label_matrix[cell_row, cell_col, class_label] = 1.0

            # -------------- OPTIONAL: If B=2 and want to store a second box --------------
            # else:
            #     # Check if second bbox slot is free (objectness at index C+5)
            #     if self.B == 2 and label_matrix[cell_row, cell_col, self.C+5] == 0:
            #         label_matrix[cell_row, cell_col, self.C+5] = 1.0
            #         label_matrix[cell_row, cell_col, self.C+6 : self.C+10] = torch.tensor(
            #             [x_cell, y_cell, w_cell, h_cell]
            #         )
            #         # class one-hot remains the same cell. Usually in YOLO v1,
            #         # class is shared for both bounding boxes.
            #     else:
            #         # If the cell is fully occupied, you might skip
            #         # or implement logic to pick whichever box has higher IoU, etc.
            #         pass

        return image, label_matrix
