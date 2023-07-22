from PIL import Image
import torch
import torchvision.transforms as transforms
from deformabletexttrans import VisionTransformer
from data import create_dataset

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


# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionTransformer(image_size, patch_size, in_channels, num_classes, embed_dim, depth, num_heads, mlp_ratio, text_rcnn_hidden_size).to(device)
model.load_state_dict(torch.load("vision_transformer_custom_dataset.pth"))
model.eval()

# Load the image and preprocess it
image_path = "Marathon/train_image/train_image/2015chongqingmls_00677.jpg"  # Replace with the path to your image
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to the input image size used during training
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)

# Perform inference
with torch.no_grad():
    bb_outputs, text_outputs = model(image_tensor)

train_dataset, val_dataset, unique_classes = create_dataset()

# Post-process the outputs (e.g., convert the text_outputs to actual text labels)
# Assuming you have the `unique_classes` list from the training code
unique_classes = train_dataset.unique_classes
text_predictions = torch.argmax(text_outputs, dim=1)
predicted_text_labels = [unique_classes[pred.item()] for pred in text_predictions]

print("Bounding Box Predictions:", bb_outputs)
print("Text Predictions:", predicted_text_labels)
