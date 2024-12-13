import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim

# Define the dataset class
class FrameDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        """
        Args:
            root_dir (str): Root directory containing the dataset.
            split (str): Dataset split to use ("train", "val", or "test").
            transform (callable, optional): Optional transforms to be applied to each sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.samples = []
        self.labels = []
        self.label_map = {
            "celeb-real": 0,
            "celeb-synthesis": 1,
            "youtube-real": 2
        }

        # Traverse through each class (e.g., celeb-real, celeb-synthesis)
        for category in os.listdir(root_dir):
            category_path = os.path.join(root_dir, category)
            if category in self.label_map:
                label = self.label_map[category]
                split_path = os.path.join(category_path, split)
                if os.path.exists(split_path):
                    for video_id in os.listdir(split_path):
                        video_path = os.path.join(split_path, video_id)
                        for frame in os.listdir(video_path):
                            frame_path = os.path.join(video_path, frame)
                            self.samples.append(frame_path)
                            self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# Define the model
class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetClassifier, self).__init__()
        self.model = models.efficientnet_b0(pretrained=True)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Main function
def main():
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset root directory
    root_dir = "frames_cropped"

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load datasets
    train_dataset = FrameDataset(root_dir=root_dir, split="train", transform=transform)
    val_dataset = FrameDataset(root_dir=root_dir, split="val", transform=transform)
    test_dataset = FrameDataset(root_dir=root_dir, split="test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Initialize model
    num_classes = 3  # Adjust based on your dataset's classes
    model = EfficientNetClassifier(num_classes=num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Initialize the best accuracy variable
    best_accuracy = 0.0

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Training Loss: {running_loss / len(train_loader):.4f}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total  # Calculate validation accuracy
        print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {val_accuracy:.2f}%")

        # Save the best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model.pth")
            print(f"New best model saved with accuracy: {best_accuracy:.2f}%")

    # Test loop
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Loss: {test_loss / len(test_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    main()
