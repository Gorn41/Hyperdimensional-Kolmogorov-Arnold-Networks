import torch
import torch.nn as nn
import torch.nn.functional as F
from torchhd.classifiers import LeHDC
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision
import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
from util.kan_convolutional.KANConv import KAN_Convolutional_Layer

class KAN(nn.Module):
    def __init__(self, grid_size: int = 5):
        super(KAN, self).__init__()
        self.conv1 = KAN_Convolutional_Layer(in_channels=3,
            out_channels= 8,
            kernel_size= (3,3),
            grid_size = grid_size
        )
        self.conv2 = KAN_Convolutional_Layer(in_channels=8,
            out_channels= 16,
            kernel_size= (3,3),
            grid_size = grid_size
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(16 * 38 * 38, 512)
        self.classifier = nn.Linear(512, 10)  # Output 10 classes for Imagenette

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.classifier(x)

        return x

def train(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}, Accuracy: {100.*correct/total:.2f}%')
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

def valtest(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
    return test_loss, test_acc

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
            outputs = model(images)
            # print(outputs)
            # print(outputs.shape)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # loss = loss_func(outputs.float(), labels)
            # total_loss += loss.item() * images.size(0)
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

            outputs = model(noisy_images)
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
    with open(f"./{name}_classification_report_noise_{noise_std}.txt", 'a', newline='') as file:
        file.truncate(0)
        file.write(f'Test Accuracy: {accuracy:.2f}%, Test Loss: {test_loss}\n')
        file.write(classification_report(all_labels, all_preds))

    return accuracy, test_loss

def load_Imagenette_data(batch_size):
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparams
    batch_size = 8
    learning_rate = 0.001
    num_epochs = 10
    
    train_loader, valloader, test_loader = load_Imagenette_data(batch_size)
    model = KAN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Lists to store metrics
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, epoch+1)
        val_loss, val_acc = valtest(model, valloader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%')

        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'KAN_baseline_results/KAN_baseline.pth')
            print(f'Model saved with val accuracy: {val_acc:.2f}%')
    
    print(f'Best val accuracy: {best_val_acc:.2f}%')
       # Plot accuracy and loss
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Val Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(epochs, val_accuracies, label='Val Accuracy', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    # Save plot
    plt.savefig('KAN_baseline_results/training_plot.png')
    plt.show()
    plt.close('all')
    model.load_state_dict(torch.load("KAN_baseline_results/KAN_baseline.pth", map_location=device, weights_only=False))
    model.eval()

    test("KAN_baseline", "KAN_baseline_results", model, test_loader, device)
    test_with_noise("KAN_baseline", "KAN_baseline_results", model, test_loader, device, noise_std=0.1)
    test_with_noise("KAN_baseline", "KAN_baseline_results",  model, test_loader, device, noise_std=0.4)
    test_with_noise("KAN_baseline", "KAN_baseline_results", model, test_loader, device, noise_std=0.7)
    test_with_noise("KAN_baseline", "KAN_baseline_results", model, test_loader, device, noise_std=1.0)

if __name__ == '__main__':
    main()
