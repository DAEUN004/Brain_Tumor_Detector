import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from utils.data_loader import BrainDataset
import os

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')


resnet50 = resnet50(weights=ResNet50_Weights.DEFAULT)
resnet50 = resnet50.to(device)
resnet50.eval()

# freeze the model weights
for param in resnet50.parameters():
    param.requires_grad = False


num_features = resnet50.fc.in_features
resnet50.fc = nn.Sequential(
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 2),  # Binary classification
    nn.Softmax(dim=1)
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet50.fc.parameters(), lr = 0.005)

class EarlyStopping:
    def __init__(self, patience=20, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered")

def save_checkpoint(state, is_best, filename="checkpoints/classifier-resnet-checkpoint.pth"):
    if is_best:
        torch.save(state, filename)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


#Get the data
dataloader = BrainDataset(csv_file = 'Healthcare_AI_Datasets/Brain_MRI/data_mask.csv', root_dir = 'Healthcare_AI_Datasets/Brain_MRI/')
train_loader, val_loader, test_loader = dataloader.get_data_loaders()


num_epochs = 10
patience = 20
early_stopping = EarlyStopping(patience=patience, verbose=True)
best_val_loss = float("inf")

for epoch in range(num_epochs):
    resnet50.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = resnet50(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

    # Validation step
    resnet50.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = resnet50(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # Save the best model checkpoint
    is_best = val_loss < best_val_loss
    best_val_loss = min(val_loss, best_val_loss)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': resnet50.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
    }, is_best)

    # Check early stopping
    early_stopping(val_loss, resnet50)
    if early_stopping.early_stop:
        print("Early stopping")
        break

torch.save(resnet50.state_dict(), 'classifier-resnet-model.pth')