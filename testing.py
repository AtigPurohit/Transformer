# from vitwvgg19andtextrcnn import MyViT
from nctransformer import MyViT
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
import numpy as np


def test_single_image(image_path, model):
    transform = ToTensor()
    image = Image.open(image_path)

    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

    # Determine the maximum dimension (height or width)
    max_dim = max(image.size[0], image.size[1])

    # Resize images to a square shape using the maximum dimension
    resizer = Resize((max_dim, max_dim))
    image = resizer(image)
    
    image = image.convert('L')  # Convert the image to grayscale

    # Resize the image to match the expected input size of the model
    image = image.resize((28, 28))

    image = transform(image).unsqueeze(0)  # Apply the transformation and add a batch dimension

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    image = image.to(device)

    model.eval()  # Set the model to evaluation mode
    # with torch.no_grad():
    #     output = model(image)


# Pass the image through the model
    with torch.no_grad():
        text_output, bounding_box = model(image)

    # Convert the tensor to numpy array
    text_output = text_output.squeeze().numpy()

# Print the textual output
    print("Text Output:", text_output)

# Print the bounding box coordinates
    print("Bounding Box:", bounding_box)    

# Load the saved model
model = MyViT((1, 28, 28), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2)
model.load_state_dict(torch.load("my_model.pt"))
model.eval()

# Test the model on a single image
image_path = "/home/spyder/Dev/Vision_Transformer_Keras/Marathon/test_image/test_image/1.jpg"
test_single_image(image_path, model)
