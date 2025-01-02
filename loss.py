import torch
import torch.nn as nn

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20, lambda_noobj = 0.5, lambda_coord = 5):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = lambda_noobj
        self.lambda_coord = lambda_coord
        
    def forward(self,predictions,target):
        """
        Calculate the loss for the model
        
        Parameters:
        predictions: tensor,  (batch_size, S, S ,(C+B*5)), predicted by the model
        target: tensor,  (batch_size, S, S, C+B*5), ground truth
        
        Returns:
        loss: float, total loss
        """
        # Split the predictions and target into components
        pred_box = predictions[..., :self.B * 5].reshape(-1, self.S, self.S, self.B, 5)
        pred_class = predictions[..., self.B * 5:]
        target_box = target[..., :self.B * 5].reshape(-1, self.S, self.S, self.B, 5)
        target_class = target[..., self.B * 5:]
        
        obj_mask = target_box[..., 4] > 0 # (batch_size, S, S, B)
        noobj_mask = ~obj_mask 
        
        # Localization loss
        box_loss = self.lambda_coord * torch.sum(obj_mask.unsqueeze(-1) * (
            (pred_box[..., :2] - target_box[..., :2])**2 +
            (torch.sqrt(torch.clamp(pred_box[..., 2:4], min=1e-6)) 
            - torch.sqrt(torch.clamp(target_box[..., 2:4], min=1e-6)))**2 # Avoid square root of negative number
        ))
        
        # Confidence loss
        conf_loss = torch.sum(
            obj_mask * (pred_box[..., 4] - target_box[..., 4])**2
        ) + self.lambda_noobj * torch.sum(
            noobj_mask * (pred_box[..., 4])**2
        )
        
        # Classification loss
        class_loss =  torch.sum(obj_mask.unsqueeze(-1) * (pred_class - target_class)**2)
        
        
        return box_loss + conf_loss + class_loss
        
        
                        
        