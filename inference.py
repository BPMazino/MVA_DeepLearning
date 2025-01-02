import torch
from preprocessing import preprocess_image, draw_bounding_boxes
from utils import non_max_suppression

def inference(model, image_path,class_names,device, checkpoint_path, nms_threshold = 0.5):
    checkpoint  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    image,_ = preprocess_image(image_path)
    image = image.to(device).unsqueeze(0)
    
    with torch.no_grad():
        predictions = model(image).squeeze(0).cpu()
    
    boxes = predictions[:, :4]
    confidences = predictions[:, 4]
    filtered_boxes = non_max_suppression(boxes, confidences, nms_threshold)
    
    draw_bounding_boxes(image.squeeze(0).cpu(), filtered_boxes, class_names)