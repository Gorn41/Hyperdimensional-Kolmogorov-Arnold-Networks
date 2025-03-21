import torch
import torch.nn as nn
import torchhd
from util.classifiers import LeHDC
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchvision
import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import csv
import pandas as pd

# Same as the CNN class in soap-models/CNN_baseline.py, except the classifier layer is removed
# This is a "template" class that can be used to create a CNN model with a custom classifier
# The custom classifier is the LeHDC classifier in this case
class CNNFeatureExtractor(nn.Module):
    def __init__(self, grid_size=5):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # Change input channels to 3 (RGB)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        self.feature_size = 64 * 20 * 20
        self.fc1 = nn.Linear(self.feature_size, 512)
        self.classifier = nn.Linear(512, 10)  # Output 100 classes for CIFAR-100
    
        # Chopped off the classifier layer


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Apply ReLU activation
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        # Chopped off the classifier layer

        return x
        
# This class is a combination of the CNNFeatureExtractor and LeHDC classes
class CNN_HDC(nn.Module):
    def __init__(self, n_dimensions=10000, n_classes=10, n_levels=100, grid_size=5):
        super(CNN_HDC, self).__init__()
        self.feature_network = CNNFeatureExtractor(grid_size=grid_size)
        
        # LeHDC classifier as a separate component
        self.lehdc = LeHDC(
            n_features=512,
            n_dimensions=n_dimensions,
            n_classes=n_classes,
            n_levels=n_levels,
            min_level=-1,
            max_level=1,
            epochs=40,
            dropout_rate=0.2,
            lr=0.005,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        self.lehdc_trained = False

    # Forward pass through the feature extractor and the LeHDC classifier
    def forward(self, x):
        features = self.feature_network.forward(x)
        return self.lehdc(features)

    def train_lehdc(self, train_loader, val_loader):
        features = []
        labels = []

        with torch.no_grad():
            for images, targets in train_loader:
                images = images.to(next(self.parameters()).device)
                feat = self.feature_network.forward(images)
                features.append(feat)
                labels.append(targets)

        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)

        dataset = torch.utils.data.TensorDataset(features.to('cuda'), labels.to('cuda'))
    
        lehdc_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=32,  
            shuffle=True   
        )

        valfeatures = []
        vallabels = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(next(self.parameters()).device)
                feat = self.feature_network.forward(images)
                valfeatures.append(feat)
                vallabels.append(targets)

        valfeatures = torch.cat(valfeatures, dim=0)
        vallabels = torch.cat(vallabels, dim=0)

        valdataset = torch.utils.data.TensorDataset(valfeatures.to('cuda'), vallabels.to('cuda'))
    
        lehdc_val_loader = torch.utils.data.DataLoader(
        valdataset, 
        batch_size=32,  
        shuffle=True   
        )

        self.lehdc.fit(lehdc_loader, lehdc_val_loader)
        self.lehdc_trained = True

def validate(model, val_loader, loss_func, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model.classify(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = loss_func(outputs, labels)
            total_loss += loss.item() * images.size(0)

    return (100 * correct / total, total_loss / len(val_loader.dataset))

def test(name, folder, model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    batch_size = len(testloader)
    loss_func = nn.CrossEntropyLoss()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm.tqdm(testloader, total=batch_size):
            images, labels = images.to(device), labels.to(device)
            outputs = model.forward(images)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted.to('cuda') == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f'Test Accuracy: {100 * correct / total:.2f}%')
    test_loss = total_loss / len(testloader.dataset)
    print(f'Test Loss: {test_loss}')
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{name} Confusion Matrix (No Noise)')
    plt.savefig(f"{folder}/{name}_confusion_matrix_no_noise.png")

    print("Classification Report:")
    print(classification_report(all_labels, all_preds))
    with open(f"{folder}/{name}_classification_report_no_noise.txt", 'a', newline='') as file:
        file.truncate(0)
        file.write(f'Test Accuracy: {100 * correct / total:.2f}%, Test Loss: {test_loss}')
        file.write(classification_report(all_labels, all_preds))
    return

def test_with_noise(name, folder, model, testloader, device, noise_std=0.1):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    loss_func = nn.CrossEntropyLoss()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            # Add Gaussian noise
            noise = torch.randn_like(images) * noise_std
            noisy_images = images + noise
            noisy_images = torch.clamp(noisy_images, 0, 1)  # Keep pixel values in [0,1]

            outputs = model.forward(noisy_images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = loss_func(outputs, labels)
            total_loss += loss.item() * images.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    test_loss = total_loss / len(testloader.dataset)

    print(f'Test Accuracy with Noise: {accuracy:.2f}%')
    print(f'Test Loss with Noise: {test_loss}')

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{name} Confusion Matrix (Noise Std = {noise_std})')
    plt.savefig(f"{folder}/{name}_confusion_matrix_noise_{noise_std}.png")
    plt.show()

    # Classification Report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds))
    with open(f"{folder}/{name}_classification_report_noise_{noise_std}.txt", 'a', newline='') as file:
        file.truncate(0)
        file.write(f'Test Accuracy: {accuracy:.2f}%, Test Loss: {test_loss}\n')
        file.write(classification_report(all_labels, all_preds))

    return accuracy, test_loss

def load_Imagenette_data(batch_size=34):
    transform = transforms.Compose([
        transforms.Resize((160, 160)),  # Resize all images to 160x160
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
    ])
    
    train_data = torchvision.datasets.Imagenette(root='data', split="train", download=True, transform=transform)
    other_data = torchvision.datasets.Imagenette(root='data', split="val", download=True, transform=transform)
    val_data, test_data = torch.utils.data.random_split(other_data, [0.5, 0.5])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    
    return train_loader, valloader, test_loader


def main():
     # Hyperparams
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10
    hdc_dimensions = 10000
    dropout_rate = 0.0
    n_levels = 150 # you can kind of think of this as rounding sensitivity

    model = CNN_HDC()
    train_loader, valloader, test_loader = load_Imagenette_data(batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.feature_network.load_state_dict(torch.load("CNN_baseline_results/CNN_baseline.pth", map_location=device))

    model.train_lehdc(train_loader, valloader)

    model.eval()

    test("CNN_HDC", "CNN_HDC_results", model, test_loader, device)
    test_with_noise("CNN_HDC", "CNN_HDC_results", model, test_loader, device, noise_std=0.1)
    test_with_noise("CNN_HDC", "CNN_HDC_results", model, test_loader, device, noise_std=0.4)
    test_with_noise("CNN_HDC", "CNN_HDC_results", model, test_loader, device, noise_std=0.7)
    test_with_noise("CNN_HDC", "CNN_HDC_results", model, test_loader, device, noise_std=1.0)

if __name__ == '__main__':
    main()