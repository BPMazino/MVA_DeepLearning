import torch
from torch.utils.data import DataLoader
from model import YOLOv1
from dataset import VOCDataset, VOCDatasetAugmented, get_predefined_splits
from loss import YoloLoss
from train import train
from validation import visualize_validation
from evaluation import test
from torchvision import transforms as T
import argparse
from inference import inference
def main():
    
    parser = argparse.ArgumentParser(description="YOLOv1 Training and Inference")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "inference"],
                        help="Mode to run: 'train' for training or 'inference' for testing and predictions.")
    parser.add_argument("--image-path", type=str, default=None,
                        help="Path to an image for inference. Required if mode is 'inference'.")
    parser.add_argument("--checkpoint-path", type=str, default="best_yolo_model.pth",
                        help="Path to the model checkpoint.")
    args = parser.parse_args()

    # Paths
    image_dir = "VOC_dataset/VOCdevkit/VOC2007/JPEGImages"
    annotation_dir = "VOC_dataset/VOCdevkit/VOC2007/Annotations"
    image_sets_dir = "VOC_dataset/VOCdevkit/VOC2007/ImageSets/Main"
    class_names = [
        "aeroplane", "bicycle", "bird", "boat", "bottle", 
        "bus", "car", "cat", "chair", "cow", "diningtable", 
        "dog", "horse", "motorbike", "person", "pottedplant", 
        "sheep", "sofa", "train", "tvmonitor"
    ]

    # Load predefined splits
    train_ids, val_ids, test_ids = get_predefined_splits(image_sets_dir)

    # Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    S = 7
    B = 2
    C = len(class_names)
    EPOCHS = 1

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transforms
    data_transform = T.Compose([
        T.Resize((448, 448)),
    ])

    # Datasets
    train_dataset = VOCDataset(image_dir, annotation_dir, class_names, image_ids=train_ids, transform=data_transform)
    val_dataset = VOCDataset(image_dir, annotation_dir, class_names, image_ids=val_ids, transform=data_transform)
    test_dataset = VOCDataset(image_dir, annotation_dir, class_names, image_ids=test_ids, transform=data_transform)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model, loss, and optimizer
    model = YOLOv1(S=S, B=B, C=C).to(device)
    loss_fn = YoloLoss(S=S, B=B, C=C).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=LEARNING_RATE, 
        momentum=MOMENTUM, 
        weight_decay=WEIGHT_DECAY
    )
    
    if args.mode == "inference":
        if not args.image_path:
            raise ValueError("For inference mode, --image-path must be provided.")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Loaded model checkpoint from {args.checkpoint_path}.")

        # Perform inference
        inference(model, args.image_path, class_names, device)
        return

    # Train the model
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=EPOCHS,
        checkpoint_path="yolo_checkpoint.pth.tar",
        device=device
    )

    # Visualize validation predictions
    print("\nVisualizing validation set predictions...")
    visualize_validation(model, val_loader, class_names, device)

    # Evaluate on the test set
    print("\nRunning evaluation on the test set...")
    test_mAP = test(model, test_loader, device)
    print(f"Final Test mAP: {test_mAP:.4f}")

if __name__ == "__main__":
    main()

