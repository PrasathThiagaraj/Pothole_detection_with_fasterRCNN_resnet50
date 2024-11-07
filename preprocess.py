import torch.multiprocessing
torch.multiprocessing.set_start_method('spawn', force=True) 

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from PIL import Image

def collate_fn(batch):
    return tuple(zip(*batch))

# Move PotholeDataset class outside the main function
class PotholeDataset(Dataset):
    def __init__(self, csv_file, img_dir, transforms=None):
        self.annotations = pd.read_csv(csv_file, header=None, names=['file_name', 'width', 'height', 'coords'])
        self.img_dir = img_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.annotations['file_name'].unique())  # Unique images count

    def __getitem__(self, idx):
        img_name = self.annotations['file_name'].unique()[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        img_data = self.annotations[self.annotations['file_name'] == img_name]

        # Collecting bounding boxes
        boxes = []
        for _, row in img_data.iterrows():
            coords = eval(row['coords'])  # Convert string coords to a list
            x_min = coords[0]
            y_min = coords[1]
            x_max = x_min + coords[2]  # x_min + width
            y_max = y_min + coords[3]  # y_min + height

            # Ensure bounding boxes have positive width and height
            if x_max > x_min and y_max > y_min:
                boxes.append([x_min, y_min, x_max, y_max])
            else:
                print(f"Invalid box found: {coords} in image {img_name}. Skipping.")

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)  # Assuming all boxes are potholes, class=1

        target = {"boxes": boxes, "labels": labels}

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target

def main():
    # Define transforms for training
    def get_transform():
        return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])

    # Creating dataset and data loader
    train_dataset = PotholeDataset(
        csv_file='C:/Users/Gaming/Desktop/my_ccp/Potholes/train.csv', 
        img_dir='C:/Users/Gaming/Desktop/my_ccp/Potholes/train', 
        transforms=get_transform()
    )

    # Update DataLoader to use the new collate function
    train_loader = DataLoader(
        train_dataset, 
        batch_size=2, 
        shuffle=True, 
        num_workers=2, 
        collate_fn=collate_fn
    )

    # Load Faster R-CNN model with pretrained weights
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Replace the classifier head with 2 classes (background, pothole)
    num_classes = 2  # 1 class for pothole + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    # Define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Training function
    def train_one_epoch(model, optimizer, data_loader, device):
        model.train()
        loss_total = 0
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backprop
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            loss_total += losses.item()

        return loss_total / len(data_loader)

    # Training loop
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    num_epochs = 10
    for epoch in range(num_epochs):
        loss = train_one_epoch(model, optimizer, train_loader, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss}")
    torch.save(model.state_dict(),"pothole_model_2.pth")

if __name__ == "__main__":
    main()
