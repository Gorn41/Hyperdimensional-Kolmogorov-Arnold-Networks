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

class LeHDCCNN(nn.Module):
    def __init__(self, hdc_dimensions=1000, n_classes=10, dropout_rate=0.1, n_levels=200):
        super(LeHDCCNN, self).__init__()
        

        self.conv1 = nn.Conv2d(1, 5, kernel_size=3)
        self.conv2 = nn.Conv2d(5, 5, kernel_size=5)
        self.conv3 = nn.Conv2d(5, 2, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        self.fc = nn.Linear(98, 256)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout_rate)

        self.lehdc = LeHDC(
            n_features=256,           
            n_dimensions=hdc_dimensions,
            n_classes=n_classes,
            n_levels=n_levels,             # you can kind of think of this as rounding
            min_level=-1,             # don't change this
            max_level=1,              # don'change this
            epochs=200,                # scale with the other epoch param
            lr=0.0001              # don't change this
        )

        
    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = self.flatten(x)
        
        x = self.fc(x)
        x = self.dropout(x)
        x = self.lehdc(x)
        
        return x

def load_mnist_data(batch_size=34):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_data = torchvision.datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
    other_data = torchvision.datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)
    val_data, test_data = torch.utils.data.random_split(other_data, [0.5, 0.5])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    
    return train_loader, valloader, test_loader

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
    plt.title('hyperparam_testing Confusion Matrix (No Noise)')
    plt.savefig("./hyperparam_testing confusion_matrix_no_noise.png")
    plt.show()

    print("Classification Report:")
    print(classification_report(all_labels, all_preds))
    with open("./hyperparam_testing_classification_report_no_noise.txt", 'a', newline='') as file:
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
    plt.title(f'hyperparam_testing Confusion Matrix (Noise Std = {noise_std})')
    plt.savefig(f"./hyperparam_testing confusion_matrix_noise_{noise_std}.png")
    plt.show()

    # Classification Report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds))
    with open(f"./hyperparam_testing_classification_report_noise_{noise_std}.txt", 'a', newline='') as file:
        # clear file
        file.truncate(0)
        file.write(f'Test Accuracy: {accuracy:.2f}%, Test Loss: {test_loss}\n')
        file.write(classification_report(all_labels, all_preds))

    return accuracy, test_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparams
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 20
    hdc_dimensions = 10000
    dropout_rate = 0.0
    n_levels = 200 # you can kind of think of this as rounding sensitivity
    
    train_loader, valloader, test_loader = load_mnist_data(batch_size)
    model = LeHDCCNN(hdc_dimensions=hdc_dimensions, n_classes=10, dropout_rate=dropout_rate, n_levels=n_levels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    accs = []
    val_losses = []
    losses = []
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, epoch+1)
        val_loss, val_acc = valtest(model, valloader, criterion, device)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%')
        accs.append(val_acc)
        val_losses.append(val_loss)
        losses.append(train_loss)
        plt.figure()
        plt.plot(np.arange(1, epoch + 2), accs)
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy")
        plt.title("LeHDC CNN Validatation Accuracy over Epochs")
        plt.savefig("./LeHDC_CNN_fashionmnist_val_acc.png")
        plt.figure()
        plt.plot(np.arange(1, epoch + 2), losses)
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title("LeHDC CNN Training Loss over Epochs")
        plt.savefig("./LeHDC_CNN_fashionmnist_training_loss.png")
        plt.figure()
        plt.plot(np.arange(1, epoch + 2), val_losses)
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.title("LeHDC CNN Baseline Validatation Loss over Epochs")
        plt.savefig("./LeHDC_CNN_fashionmnist_val_loss.png")
        plt.close('all')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'cnn_lehdc_best.pth')
            print(f'Model saved with val accuracy: {val_acc:.2f}%')
    
    print(f'Best val accuracy: {best_val_acc:.2f}%')


    model.load_state_dict(torch.load("cnn_lehdc_best.pth", map_location=device, weights_only=False))

    model.eval()

    test(model, test_loader, device)
    test_with_noise(model, test_loader, device, noise_std=0.1)
    test_with_noise(model, test_loader, device, noise_std=0.4)
    test_with_noise(model, test_loader, device, noise_std=0.7)
    test_with_noise(model, test_loader, device, noise_std=1.0)

if __name__ == '__main__':
    main()
