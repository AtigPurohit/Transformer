import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torch.utils.data import random_split


class CustomDataset(Dataset):
    def __init__(self, image_folder, label_folder, transform=None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.image_list = os.listdir(image_folder)
        self.texts = []  # List to store the class labels (texts)

        # Extract and store the class labels (texts) during initialization
        for image_name in self.image_list:
            label_filename = os.path.splitext(image_name)[0] + ".txt"
            label_path = os.path.join(self.label_folder, label_filename)
            with open(label_path, "r") as f:
                label_values = f.readline().strip().split()
            text = label_values[-1]  # Get the last value as the class label (text)
            self.texts.append(text)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert("RGB")

        # Load label from text file
        label_filename = os.path.splitext(image_name)[0] + ".txt"
        label_path = os.path.join(self.label_folder, label_filename)
        with open(label_path, "r") as f:
            label_values = f.readline().strip().split()

        # Extract bounding box coordinates and text from the label
        num_coords = len(label_values)
        bbox_coords = list(map(int, label_values[:num_coords-2]))
        label = int(label_values[num_coords-2])  # Convert label to integer
        text = label_values[num_coords-1]

        # Convert bounding box to (x_min, y_min, x_max, y_max) format and normalize
        width, height = image.size
        num_points = len(bbox_coords) // 2
        x_coords = bbox_coords[:num_points]
        y_coords = bbox_coords[num_points:]
        x_min, y_min = min(x_coords), min(y_coords)
        x_max, y_max = max(x_coords), max(y_coords)
        bbox = torch.tensor([x_min / width, y_min / height, x_max / width, y_max / height], dtype=torch.float)

        if self.transform:
            image = self.transform(image)

        return image, bbox, label, text


    
def create_dataset():
    # Data augmentation and normalization
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    image_folder = "Marathon/train_image/train_image"
    label_folder = "Marathon/train_gt3"

    dataset = CustomDataset(image_folder=image_folder, label_folder=label_folder, transform=transform)

    # Extract unique class names (texts) from the dataset
    unique_classes = list(set(dataset.texts))

    # Split dataset into train and validation subsets
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)  # 80% for training
    val_size = dataset_size - train_size  # Remaining 20% for validation

    # Perform random split and retrieve the texts for each subset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataset.texts = [dataset.texts[i] for i in train_dataset.indices]
    val_dataset.texts = [dataset.texts[i] for i in val_dataset.indices]

    return train_dataset, val_dataset, unique_classes