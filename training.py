from tqdm import tqdm, trange
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.nn import CrossEntropyLoss
from nctransformer import MyViT
import torchvision.datasets as datasets
import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import cv2
from sklearn.model_selection import train_test_split


class CustomDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_paths, self.labels = self._load_data()
        self.transform = transform

    def _load_data(self):
        image_paths = []
        labels = []

        # Load image paths
        for filename in os.listdir(self.image_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(self.image_dir, filename)
                image_paths.append(image_path)

        # Load labels
        for filename in os.listdir(self.label_dir):
            if filename.endswith('.txt'):
                label_path = os.path.join(self.label_dir, filename)
                with open(label_path, 'r') as f:
                    label = f.read().strip()
                    labels.append(label)

        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)

        return image, label

def main():
    # Define image transforms
    transform = ToTensor()

    subset_ratio = 0.5
    # Define directories
    image_dir = '/home/spyder/Dev/Vision_Transformer_Keras/Marathon/train_image/train_image'
    label_dir = '/home/spyder/Dev/Vision_Transformer_Keras/Marathon/train_gt/train_gt'

    dataset = CustomDataset(image_dir, label_dir, transform=transform)

    # Perform train-test split
    test_size = 0.2
    random_state = 42
    train_dataset, test_dataset = train_test_split(dataset, test_size=test_size, random_state=random_state)

    # Create the data loaders
    batch_size = 16
    train_subset = train_dataset[:int(subset_ratio * len(train_dataset))]
    test_subset = test_dataset[:int(subset_ratio * len(test_dataset))]

    num_workers = 4
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,num_workers=num_workers)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False,num_workers=num_workers)

     # Define the model and training options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyViT((1, 28, 28), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2).to(device)
    N_EPOCHS = 1
    LR = 0.005

    # Training loop
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss()
    for epoch in trange(N_EPOCHS, desc="Training"):
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

    # Test loop
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
            
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")

    # Save the model
    torch.save(model.state_dict(), "my_model.pt")
    print("Model saved.")



if __name__ == "__main__":
    main()