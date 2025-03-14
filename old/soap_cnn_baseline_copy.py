import torch
import torch.nn as nn
import torch.nn.functional as F
import torchhd
from torchhd.classifiers import LeHDC
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
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
import deeplake

class CNN(nn.Module):
    def __init__(self, n_classes=1000, dropout_rate=0.1):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        # Let's calculate the correct input size for the fully connected layer
        # Original input: 32x32
        # After pool in forward: 16x16
        # After second convolution: still 16x16 (due to padding=1)
        # After third convolution: still 16x16 (due to padding=1)
        # So feature map size is 128 x 16 x 16 = 32768
        self.fc = nn.Linear(32768, 512)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, n_classes)

        
    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = self.flatten(x)
        
        x = self.fc(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
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

class ImageNet32Dataset(Dataset):
    def __init__(self, npz_paths, transform=None):
        self.images = []
        self.labels = []
        
        for npz_path in npz_paths:
            data = np.load(npz_path, allow_pickle=True)
            self.images.append(data['data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0)
            # Subtract 1 from labels if they are 1-indexed (1 to 1000)
            # Or use modulo to ensure they're in the right range
            labels = np.array(data['labels'], dtype=np.int64) % 1000
            self.labels.append(labels)
        
        self.images = np.concatenate(self.images, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        
        # Print shape for debugging
        # print(f"Original image shape: {img.shape}")
        
        # Convert numpy array to tensor
        img = torch.from_numpy(img)
        
        # Print tensor shape
        # print(f"Tensor shape: {img.shape}")
        
        # Apply transformations
        if self.transform:
            img = self.transform(img)
            # print(f"After transform: {img.shape}")
            
        return img, label

    
def load_imagenet32(batch_size=64, train_npz_paths=[], val_npz=''):
    # Only include normalization since we're handling tensor conversion in the dataset
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = ImageNet32Dataset(train_npz_paths, transform=transform)
    val_dataset, test_dataset = torch.utils.data.random_split(
        ImageNet32Dataset([val_npz], transform=transform), [0.5, 0.5])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader

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
    plt.title('soap_cnn_imagenet Confusion Matrix (No Noise)')
    plt.savefig("./soap_cnn_imagenet confusion_matrix_no_noise.png")
    plt.show()

    print("Classification Report:")
    print(classification_report(all_labels, all_preds))
    with open("./soap_cnn_imagenet_classification_report_no_noise.txt", 'a', newline='') as file:
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
    plt.title(f'soap_cnn_imagenet Confusion Matrix (Noise Std = {noise_std})')
    plt.savefig(f"./soap_cnn_imagenet confusion_matrix_noise_{noise_std}.png")
    plt.show()

    # Classification Report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds))
    with open(f"./soap_cnn_imagenet_classification_report_noise_{noise_std}.txt", 'a', newline='') as file:
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
    hdc_dimensions = 10000
    dropout_rate = 0.0
    n_levels = 150 # you can kind of think of this as rounding sensitivity
    
    train_npz_paths = [f'imagenet/Imagenet32_train/train_data_batch_{i}' for i in range(1, 11)]
    val_npz = 'imagenet/val_data'
    
    train_loader, val_loader, test_loader = load_imagenet32(batch_size, train_npz_paths, val_npz)
    model = CNN(n_classes=1000).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, epoch+1)
        val_loss, val_acc = valtest(model, val_loader, criterion, device)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'imagenet32_cnn_best.pth')
            print(f'Model saved with val accuracy: {val_acc:.2f}%')
    
    print(f'Best val accuracy: {best_val_acc:.2f}%')

    model.load_state_dict(torch.load("imagenet32_cnn_best.pth", map_location=device, weights_only=False))

    model.eval()

    test(model, test_loader, device)
    test_with_noise(model, test_loader, device, noise_std=0.1)
    test_with_noise(model, test_loader, device, noise_std=0.4)
    test_with_noise(model, test_loader, device, noise_std=0.7)
    test_with_noise(model, test_loader, device, noise_std=1.0)

if __name__ == '__main__':
    main()
