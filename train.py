import torch
import os
from utils import compute_map 
from PIL import Image
def save_checkpoint(epoch, model, optimizer, filename="checkpoint.pth"):
    """
    Save the model and optimizer state
    
    Parameters:
    model: torch model
    optimizer: torch optimizer
    filename: str, name of the checkpoint file
    
    Returns:
    None
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)
    
def load_checkpoint(model, optimizer ,filename="checkpoint.pth"):
    """
    Load the model and optimizer state
    
    Parameters:
    model: torch model
    filename: str, name of the checkpoint file
    
    Returns:
    start_epoch: int, starting epoch
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    return start_epoch
    
def adjusting_learning_rate(optimizer, epoch, initial_lr):
    """
    Adjust the learning rate during training
    
    Parameters:
    optimizer: torch optimizer
    epoch: int, current epoch
    initial_lr: float, initial learning rate
    
    Returns:
    None
    """
    if epoch < 5: # Warm-up
        lr = initial_lr * (epoch + 1) / 5
    elif epoch < 80:
        lr = initial_lr
    elif epoch < 110:
        lr = initial_lr * 0.1
    else:
        lr = initial_lr * 0.01
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
        
        


def train(model, train_loader, val_loader,loss_fn, optimizer, epochs, device,checkpoint_path = "checkpoint.pth"):
    """
    Train the model
    
    Parameters:
    model: torch model, YOLOv1
    loss_fn: loss function
    optimizer: torch optimizer
    train_loader: torch DataLoader
    epochs: int, number of epochs
    
    Returns:
    None
    """
    best_map = 0
    model.to(device)
    initial_lr = optimizer.param_groups[0]['lr']
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
        print(f"Model checkpoint loaded at epoch {start_epoch}")
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0
        
        
        current_lr = adjusting_learning_rate(optimizer, epoch, initial_lr)
        print(f"Epoch [{epoch + 1}/{epochs}], Current Learning Rate: {current_lr:.6f}")

        
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}")
        
        val_loss, mAP = validate(model, val_loader, loss_fn, device)

        # Save the best model based on mAP
        if mAP > best_mAP:
            best_mAP = mAP
            torch.save(model.state_dict(), "best_yolo_model.pth")
            print(f"New best model saved with mAP: {mAP:.4f}")
        
        if (epoch + 1) % 10 == 0:
            save_checkpoint(epoch, model, optimizer,checkpoint_path = f"yolo_checkpoint_{epoch + 1}.pth")
            print(f"Model checkpoint saved at epoch {epoch + 1}")

            
def validate(model, val_loader, loss_fn, device, iou_threshold=0.5):
    """
    Validate the model on the validation set
    
    Parameters:
    model: torch model, trained YOLOv1
    val_loader: torch DataLoader, validation set
    loss_fn: loss function
    device : torch device
    iou_threshold: float, Intersection over Union (IoU) threshold
    
    Returns:
    val_loss: float, validation loss
    mAP: float, Mean Average Precision
    """
    
    model.eval()
    total_loss = 0
    all_predictions = []
    all_ground_truths = []



    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)

            predictions = model(images)
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()

            all_predictions.append(predictions.cpu())
            all_ground_truths.append(targets.cpu())

    if not all_predictions or not all_ground_truths:
        print("No predictions or ground truths found in validation.")
        return float('inf'), 0.0  # Return high loss and low mAP

    mAP = compute_map(all_predictions, all_ground_truths, iou_threshold)
    val_loss = total_loss / len(val_loader)

    print(f"Validaiton Loss : {val_loss:.4f}, mAP@{iou_threshold} : {mAP:.4f}")
    return val_loss, mAP       

    
