import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torchvision.ops.deform_conv import DeformConv2d
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torch.utils.data import random_split
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from torchvision.models import vgg19
import torchvision.models as models

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H == self.image_size[0] and W == self.image_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model's expected size ({self.image_size[0]}*{self.image_size[1]})."
        x = self.projection(x).flatten(2).transpose(1, 2)  # B, P*P, E
        return x

class DeformableMLPHead(nn.Module):
    def __init__(
        self, in_features, hidden_features, num_classes
    ):
        super(DeformableMLPHead, self).__init__()
        self.deformable_conv1 = DeformConv2d(
            in_features, hidden_features, kernel_size=3, padding=1
        )
        self.deformable_conv2 = DeformConv2d(
            hidden_features, hidden_features, kernel_size=3, padding=1
        )
        self.fc = nn.Linear(hidden_features, num_classes)

    def forward(self, x, offset):
        x = F.relu(self.deformable_conv1(x, offset))
        x = F.relu(self.deformable_conv2(x, offset))
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        output = self.fc(x)
        return output


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        in_channels,
        num_classes,
        embed_dim,
        depth,
        num_heads,
        mlp_ratio,
    ):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(
            image_size, patch_size, in_channels, embed_dim
        )
        self.patch_embed.requires_grad_(True)
        self.image_size = image_size  # Define the image size attribute
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(0.1)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                embed_dim, num_heads, int(embed_dim * mlp_ratio)
            ),
            depth,
        )

        # Initialize the channel adapter
        self.channel_adapter = nn.Conv2d(in_channels, 3, kernel_size=1)

        # Initialize VGG19
        self.vgg = models.vgg19(pretrained=True)
        vgg_out_features = self.vgg.classifier[-1].in_features
        self.vgg.classifier[
            -1
        ] = nn.Identity()  # Remove the final fully connected layer

        self.offset_layer = nn.Conv2d(
            embed_dim, 18, kernel_size=3, padding=1
        )  # 18 for 2D offsets (2*kernel_size^2)

        self.deformable_mlp_head = DeformableMLPHead(
            embed_dim, 256, num_classes
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize transformer weights
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

        # Initialize Deformable Convolution weights
        for name, param in self.named_parameters():
            if "offset_layer" in name:
                nn.init.normal_(param, std=0.02)
            elif "deformable_mlp_head" in name:
                if "weight" in name:
                    nn.init.normal_(param, std=0.02)
                elif "bias" in name:
                    nn.init.zeros_(param)

    def forward(self, x):
        B = x.shape[0]

        # Use 1x1 convolutional layer for channel adaptation
        x = self.channel_adapter(x)

        # Use VGG19 as a feature extractor
        vgg_features = self.vgg(x)  # Output shape: (B, C, H', W')

        # Add an extra dimension for adaptive_avg_pool2d to work correctly
        vgg_features = vgg_features.unsqueeze(2)

        # Calculate the number of patches in the Vision Transformer
        num_patches = (self.image_size[0] // self.patch_size) * (
            self.image_size[1] // self.patch_size
        )

        # Resize VGG features to match the ViT input size
        vgg_features = F.adaptive_avg_pool2d(
            vgg_features,
            (
                self.image_size[0] // self.patch_size,
                self.image_size[1] // self.patch_size,
            ),
        )
        vgg_features = vgg_features.view(B, num_patches, -1)  # Reshape to (B, P*P, E)

        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)

        x = self.transformer_encoder(x)

        # Extract cls_token representation
        x = x[:, 0]

        # Reshape x to (B, C, H, W) before passing to the offset_layer
        x = x.view(B, -1, 1, 1)

        # Compute offsets for deformable convolution
        offset = self.offset_layer(x)

        # Predict bounding box coordinates using deformable
        bb_predictions = self.deformable_mlp_head(x, offset)

        return bb_predictions
    
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
        bbox_coords = list(map(int, label_values[: num_coords - 2]))
        label = int(label_values[num_coords - 2])  # Convert label to integer
        text = label_values[num_coords - 1]

        # Convert bounding box to (x_min, y_min, x_max, y_max) format and normalize
        width, height = image.size
        num_points = len(bbox_coords) // 2
        x_coords = bbox_coords[:num_points]
        y_coords = bbox_coords[num_points:]
        x_min, y_min = min(x_coords), min(y_coords)
        x_max, y_max = max(x_coords), max(y_coords)
        bbox = torch.tensor(
            [x_min / width, y_min / height, x_max / width, y_max / height],
            dtype=torch.float,
        )

        if self.transform:
            image = self.transform(image)

        return image, bbox, label, text


def create_dataset():
    # Data augmentation and normalization
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    image_folder = "Marathon/train_image/train_image"
    label_folder = "Marathon/train_gt3"

    dataset = CustomDataset(
        image_folder=image_folder, label_folder=label_folder, transform=transform
    )

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


def training():
    # Define training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    batch_size = 128
    num_epochs = 1
    learning_rate = 1e-3
    weight_decay = 1e-4

    # Data augmentation and normalization
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataset, val_dataset, unique_classes = create_dataset()
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # Initialize the Vision Transformer model
    image_size = (32, 32)  # CIFAR-10 image size
    patch_size = 8  # Smaller patch size to reduce computation
    in_channels = 3
    num_classes = 4  # Assuming 4 classes for text detection
    embed_dim = 256
    depth = 12
    num_heads = 4
    mlp_ratio = 4

    model = VisionTransformer(
        image_size,
        patch_size,
        in_channels,
        num_classes,
        embed_dim,
        depth,
        num_heads,
        mlp_ratio,
    ).to(device)

    # Define the loss function for bounding box predictions
    bb_criterion = nn.SmoothL1Loss()

    # Define the optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        total = 0

        with tqdm(
            total=len(train_loader), desc=f"Epoch {epoch}/{num_epochs}", unit="batch"
        ) as pbar:
            for batch_idx, (data, bbox_coords, labels, texts) in enumerate(
                train_loader
            ):
                data, bbox_coords, labels, texts = (
                    data.to(device),
                    bbox_coords.to(device),
                    labels.to(device),
                    texts,
                )

                optimizer.zero_grad()
                bb_outputs = model(data)

                # Flatten both bbox_coords and bb_outputs
                batch_size, num_bboxes = bbox_coords.size()
                num_classes = bb_outputs.size(1)

                bbox_coords = bbox_coords.view(
                    -1, 4
                )  # Reshape to [batch_size * num_bboxes, 4]
                bb_outputs = bb_outputs.view(
                    -1, num_classes
                )  # Reshape to [batch_size * num_bboxes, num_classes]

                # Compute bounding box loss
                bb_loss = bb_criterion(bb_outputs, bbox_coords)

                loss = bb_loss
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                total += data.size(0)

                # Update the progress bar
                pbar.set_postfix({"Loss": train_loss / (batch_idx + 1)})
                pbar.update()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, bbox_coords, labels, texts in val_loader:
                data, bbox_coords = data.to(device), bbox_coords.to(device)

                bb_outputs = model(data)
                bb_loss = bb_criterion(bb_outputs, bbox_coords)

                val_loss += bb_loss.item()

        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss}")

        # Adjust learning rate
        scheduler.step()

    print("Training finished!")

    # Save the trained model
    torch.save(model.state_dict(), "reidentification_model.pth")

if __name__ == '__main__':
    training()
