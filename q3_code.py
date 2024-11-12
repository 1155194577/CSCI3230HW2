import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Define the transformations
transform = transforms.Compose([
    transforms.Resize((150, 150)),  # Resize images
    transforms.ToTensor(),           # Convert images to tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
])

# 2. Custom Dataset Class
class CatDogDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

# 3. Load the training dataset
train_data_dir = './q3_data/q3_data/train'  # Directory for training data
image_paths_train = []
labels_train = []

# Load images from 'cat' folder in training set
cat_train_folder = os.path.join(train_data_dir, 'cat')
for img_file in os.listdir(cat_train_folder):
    image_paths_train.append(os.path.join(cat_train_folder, img_file))
    labels_train.append(0)  # Label for cat

# Load images from 'dog' folder in training set
dog_train_folder = os.path.join(train_data_dir, 'dog')
for img_file in os.listdir(dog_train_folder):
    image_paths_train.append(os.path.join(dog_train_folder, img_file))
    labels_train.append(1)  # Label for dog

# Convert to numpy arrays
image_paths_train = np.array(image_paths_train)
labels_train = np.array(labels_train)

# Create training dataset and dataloader
train_dataset = CatDogDataset(image_paths_train, labels_train, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 4. Load the testing dataset
test_data_dir = './q3_data/q3_data/test'  # Directory for testing data
image_paths_test = []
labels_test = []

# Load images from 'cat' folder in testing set
cat_test_folder = os.path.join(test_data_dir, 'cat')
for img_file in os.listdir(cat_test_folder):
    image_paths_test.append(os.path.join(cat_test_folder, img_file))
    labels_test.append(0)  # Label for cat

# Load images from 'dog' folder in testing set
dog_test_folder = os.path.join(test_data_dir, 'dog')
for img_file in os.listdir(dog_test_folder):
    image_paths_test.append(os.path.join(dog_test_folder, img_file))
    labels_test.append(1)  # Label for dog

# Convert to numpy arrays
image_paths_test = np.array(image_paths_test)
labels_test = np.array(labels_test)

# Create testing dataset and dataloader
test_dataset = CatDogDataset(image_paths_test, labels_test, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 5. [Task 1]: Define the CNN model
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 18 * 18, 512)
        self.fc2 = nn.Linear(512, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 18 * 18)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)

# 6. Initialize the model, define loss function and optimizer.
model = MyCNN().to(device)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 7. [Task 2] Train the model and store loss and accuracy values
num_epochs = 10  # Set the total number of epochs
train_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        labels = labels.float().view(-1, 1).to(device)  # Reshape labels for BCELoss
        images = images.to(device)  # Move images to device
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(images)  # Forward pass 
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        running_loss += loss.item() 
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss) 

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    # [Task 3] Evaluate every 10 epochs
    if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                labels = labels.float().view(-1, 1).to(device)  # Reshape labels for BCELoss
                images = images.to(device)  # Move images to device
                
                outputs = model(images)  # Forward pass

                # Get predictions
                predicted = torch.round(outputs)  # No need to apply sigmoid again
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        val_accuracies.append(accuracy)
        print(f'Test accuracy of the model every 10 epochs: {accuracy:.2f}%')

# 8. Plot the training loss
plt.figure(figsize=(10, 5))
print("train_losses", train_losses)
ep_vector = np.array(range(1, num_epochs + 1))
train_vector = np.array(train_losses)
plt.plot(ep_vector, train_vector, marker='o', label='Training Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(range(1, num_epochs + 1))
plt.legend()
plt.show()
plt.savefig("loss.png")

# 9. Plot the test accuracy
plt.figure(figsize=(10, 5))
plt.plot(ep_vector, np.array(val_accuracies), marker='o', label='Test Accuracy', color='orange')
print("accuracy:", val_accuracies)
plt.title('Test Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.xticks(range(1, num_epochs + 1))
plt.legend()
plt.show()
plt.savefig("acc.png")