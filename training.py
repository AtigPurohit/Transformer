import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from deformabletexttrans import VisionTransformer
from data import create_dataset

#  Define training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
# device = "cpu"
# print("Using device: ", device)
batch_size = 128
num_epochs = 50
learning_rate = 1e-3
weight_decay = 1e-4

# Data augmentation and normalization
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset, val_dataset, unique_classes = create_dataset()
# # Load CIFAR-10 dataset
# cifar_train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
# train_size = int(0.8 * len(cifar_train_dataset))
# val_size = len(cifar_train_dataset) - train_size
# train_dataset, val_dataset = random_split(cifar_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Initialize the model and optimizer
image_size = (32, 32)  # CIFAR-10 image size
patch_size = 8  # Smaller patch size to reduce computation
in_channels = 3
num_classes = 9000  # CIFAR-10 has 10 classes
embed_dim = 256
depth = 12
num_heads = 4
mlp_ratio = 4
text_rcnn_hidden_size = 128

model = VisionTransformer(image_size, patch_size, in_channels, num_classes, embed_dim, depth, num_heads, mlp_ratio, text_rcnn_hidden_size).to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)


# Define the loss functions for bounding box predictions and text recognition predictions
bb_criterion = nn.CrossEntropyLoss()
text_criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    bb_correct = 0
    text_correct = 0
    total = 0

    with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{num_epochs}", unit="batch") as pbar:
        for batch_idx, (data, bbox_coords, labels, texts) in enumerate(train_loader):
            data, bbox_coords, labels, texts = data.to(device), bbox_coords.to(device), labels.to(device), texts

            # Create a mapping from text to numerical indices
            text_to_idx = {text: idx for idx, text in enumerate(unique_classes)}

            # Convert the text labels to numerical indices using the text_to_idx mapping
            text_indices = [text_to_idx[text] for text in texts]

            # Now convert the numerical indices to a tensor
            text_targets = torch.tensor(text_indices, dtype=torch.long, device=device)

            
            optimizer.zero_grad()
            bb_outputs, text_outputs = model(data)

            #  # Flatten both bbox_coords and bb_outputs
            batch_size, num_bboxes = bbox_coords.size()
            num_classes = bb_outputs.size(1)

            bbox_coords = bbox_coords.view(-1, 4)  # Reshape to [batch_size * num_bboxes, 4]
            bb_outputs = bb_outputs.view(-1, num_classes)  # Reshape to [batch_size * num_bboxes, num_classes]

            # # Compute bounding box loss
            # bb_loss = nn.SmoothL1Loss()(bb_outputs, bbox_coords)

            # Compute text recognition loss (assuming you have a text_criterion for text recognition task)
            text_loss = text_criterion(text_outputs, text_targets)

            # Combined loss (you can adjust the weighting of the losses if needed)
            # loss = bb_loss + text_loss
            loss = text_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # For accuracy calculation, you need to convert text_outputs to predictions (if needed)
            # and then compare with text_targets
            # bb_correct and text_correct can be updated accordingly

            # Update the progress bar
            pbar.set_postfix({
              "Loss": train_loss / (batch_idx + 1),
              # "BB Acc": 100. * bb_correct / total,
              # "Text Acc": 100. * text_correct / total
              })
            pbar.update()


    # # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad(), tqdm(total=len(val_loader), desc="Validation", unit="batch") as pbar:
        for data, bbox_coords, labels, texts in val_loader:
            data, bbox_coords, labels = data.to(device), bbox_coords.to(device), labels.to(device)
            bb_outputs, text_outputs = model(data)

            _, bb_predicted = torch.max(bb_outputs.data, 1)
            _, text_predicted = torch.max(text_outputs.data, 1)

            total += bbox_coords.size(0)
            bb_correct += (bb_predicted == bbox_coords).sum().item()

            # Create a mapping from text to numerical indices
            text_to_idx = {text: idx for idx, text in enumerate(unique_classes)}

            # Convert the text labels to numerical indices using the text_to_idx mapping
            text_indices = [text_to_idx[text] for text in texts]

            # Now convert the numerical indices to a tensor
            text_targets = torch.tensor(text_indices, dtype=torch.long, device=device)

            text_correct += (text_predicted == text_targets).sum().item()

            # Update the tqdm bar
            pbar.update()


    # val_bb_accuracy = 100. * bb_correct / total
    val_text_accuracy = 100. * text_correct / total
    print(f"Validation BB Accuracy: , Validation Text Accuracy: {val_text_accuracy:.2f}%")

print("Training finished!")

# Save the trained model
torch.save(model.state_dict(), "vision_transformer_custom_dataset.pth")