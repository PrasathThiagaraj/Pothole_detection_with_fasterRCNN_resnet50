import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
from PIL import Image
import os
from torchvision import models

# Load the model for inference
def load_model(model_path, device):
    # Define the model architecture with 2 classes (background and pothole)
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    
    # Replace the box predictor to have 2 classes (background, pothole)
    num_classes = 2  # 1 class for pothole + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    # Load the saved state_dict into the model
    state_dict = torch.load(model_path, map_location=device)

    # Load the weights into the model
    model.load_state_dict(state_dict)

    # Move the model to the specified device (GPU or CPU)
    model.to(device)
    model.eval()

    return model


# Function to visualize images with bounding boxes
def visualize_prediction(image, boxes, labels):
    # Convert the numeric labels (tensor) to string labels
    str_labels = [f'Class {label.item()}' for label in labels]

    # Draw bounding boxes on the image
    image = F.to_tensor(image)  # Convert PIL image to tensor
    image_with_boxes = draw_bounding_boxes(image, boxes, labels=str_labels, colors="red", width=2)

    # Convert back to numpy for displaying using matplotlib
    image_with_boxes = image_with_boxes.permute(1, 2, 0).cpu().numpy()
    plt.imshow(image_with_boxes)
    plt.axis('off')
    plt.show()


# Function to load and preprocess images
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Open image as RGB
    return image

# Function to run inference on an image
def predict(model, image, device):
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)  # Convert to tensor and add batch dimension
    with torch.no_grad():  # No gradients needed for inference
        prediction = model(image_tensor)[0]  # Get the first result
    return prediction

# Main testing/deployment function
def test_model_on_images(model, test_img_dir, device):
    for img_name in os.listdir(test_img_dir):
        img_path = os.path.join(test_img_dir, img_name)
        image = load_image(img_path)

        # Make prediction
        prediction = predict(model, image, device)

        # Extract predicted boxes, labels, and scores
        pred_boxes = prediction['boxes'].cpu()
        pred_labels = prediction['labels'].cpu()
        pred_scores = prediction['scores'].cpu()

        # Visualize the prediction on the image
        print(f"Predicted bounding boxes for {img_name}:")
        print(pred_boxes)  # Or you can log it
        visualize_prediction(image, pred_boxes, pred_labels)

# Main function
def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Path to the trained model weights
    model_path = 'C:/Users/Gaming/Desktop/my_ccp/pothole_model_2.pth'

    # Load the trained model
    model = load_model(model_path, device)

    # Directory containing test images (unlabeled)
    test_img_dir = 'C:/Users/Gaming/Desktop/my_ccp/Potholes/test'  # No CSV needed here, just a folder of images

    # Run the model on test images
    test_model_on_images(model, test_img_dir, device)

if __name__ == '__main__':
    main()
