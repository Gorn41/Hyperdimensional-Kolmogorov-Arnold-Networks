import torch
import torch.nn as nn
import torch.nn.functional as F
import torchhd
from torchhd.classifiers import LeHDC
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision
import tqdm
import matplotlib.pyplot as plt
import copy
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import csv
import pandas as pd
from kan_convolutional.KANConv import KAN_Convolutional_Layer

class CNN(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 5, kernel_size=3) 
        self.conv2 = nn.Conv2d(5, 5, kernel_size=3)
        self.conv3 = nn.Conv2d(5, 15, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        self._to_linear = None
        self._compute_conv_output_size()

        self.fc = nn.Linear(self._to_linear, 750)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(750, 10)

    def _compute_conv_output_size(self):
        with torch.no_grad():
            sample = torch.randn(1, 3, 32, 32) 
            sample = self.pool(F.relu(self.conv1(sample)))
            sample = F.relu(self.conv2(sample))
            sample = F.relu(self.conv3(sample))
            self._to_linear = sample.view(1, -1).size(1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def load_stl10_data(batch_size=32, val_split=0.2):
    transform_stl10 = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to 32x32
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    
    stl10_train = datasets.STL10(root='data', split='train', download=True, transform=transform_stl10)
    stl10_test = datasets.STL10(root='data', split='test', download=True, transform=transform_stl10)
    
    stl10_train, stl10_val = torch.utils.data.random_split(stl10_train, [int((1-val_split) * len(stl10_train)), int(val_split * len(stl10_train))])
    
    train_loader = DataLoader(stl10_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(stl10_val, batch_size=batch_size)
    test_loader = DataLoader(stl10_test, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader


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

def test(model, testloader, device):
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
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
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
    plt.title('STL10 Confusion Matrix (No Noise)')
    plt.savefig("./STL10_results/STL10_baselineCNN_confusion_matrix_no_noise.png")
    plt.show()

    print("Classification Report:")
    print(classification_report(all_labels, all_preds))
    with open("./STL10_results/STL10_baselineCNN_classification_report_no_noise.txt", 'a', newline='') as file:
        file.write(f'Test Accuracy: {100 * correct / total:.2f}%, Test Loss: {test_loss}')
        file.write(classification_report(all_labels, all_preds))
    return

def test_with_noise(model, testloader, device, noise_std=0.1):
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
    plt.title(f'STL10 Confusion Matrix (Noise Std = {noise_std})')
    plt.savefig(f"./STL10_results/STL10_baselineCNN_confusion_matrix_noise_{noise_std}.png")
    plt.show()

    # Classification Report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds))
    with open(f"./STL10_results/STL10_baselineCNN_classification_report_noise_{noise_std}.txt", 'a', newline='') as file:
        # clear file
        file.truncate(0)
        file.write(f'Test Accuracy: {accuracy:.2f}%, Test Loss: {test_loss}\n')
        file.write(classification_report(all_labels, all_preds))

    return accuracy, test_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparams
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10
    dropout_rate = 0.0
    
    train_loader, val_loader, test_loader = load_stl10_data(batch_size)
    model = CNN(dropout_rate=dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_acc = 0
    accs, val_losses, losses = [], [], []
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, epoch+1)
        val_loss, val_acc = valtest(model, val_loader, criterion, device)
        
        accs.append(val_acc)
        val_losses.append(val_loss)
        losses.append(train_loss)
        
        plt.figure()
        plt.plot(np.arange(1, epoch + 2), accs)
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy")
        plt.title("LeHDC CNN Validation Accuracy over Epochs")
        plt.savefig("./STL10_results/STL10_baselineCNN_val_acc.png")
        
        plt.figure()
        plt.plot(np.arange(1, epoch + 2), losses)
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title("LeHDC CNN Training Loss over Epochs")
        plt.savefig("./STL10_results/STL10_baselineCNN_training_loss.png")
        
        plt.figure()
        plt.plot(np.arange(1, epoch + 2), val_losses)
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.title("LeHDC CNN Baseline Validation Loss over Epochs")
        plt.savefig("./STL10_results/STL10_baselineCNN_val_loss.png")
        
        plt.close('all')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), './STL10_results/STL10_baselineCNN_best.pth')
            print(f'Model saved with val accuracy: {val_acc:.2f}%')
    
    print(f'Best val accuracy: {best_val_acc:.2f}%')
    
    model.load_state_dict(torch.load("./STL10_results/STL10_baselineCNN_best.pth", map_location=device))
    model.eval()
    
    test(model, test_loader, device)
    test_with_noise(model, test_loader, device, noise_std=0.1)
    test_with_noise(model, test_loader, device, noise_std=0.4)
    test_with_noise(model, test_loader, device, noise_std=0.7)
    test_with_noise(model, test_loader, device, noise_std=1.0)

if __name__ == '__main__':
    main()
