import torch
import torch.nn as nn
import torchhd
from classifiers import LeHDC
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
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
from kan_convolutional.KANConv import KAN_Convolutional_Layer

class KANCFeatureExtractor(nn.Module):
    def __init__(self, grid_size=5):
        super(KANCFeatureExtractor, self).__init__()
        self.flat = nn.Flatten()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 5, 3),
            
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(5, 10, 3),

            nn.PReLU(),

            nn.Conv2d(10, 15, 3)
        )
        
        self.feature_size = 1215
        self.fc = nn.Linear(self.feature_size, 750)
        self.classifier = nn.Linear(750, 10)  # Default to 10 classes
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()


    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.flat(x)
        x = self.fc(x)
        return x

    def classify(self, x):
        features = self.forward(x)
        return self.classifier(features)

class KANCLeHDCModel(nn.Module):
    def __init__(self, n_dimensions=10000, n_classes=10, n_levels=50, grid_size=5):
        super(KANCLeHDCModel, self).__init__()
        self.feature_network = KANCFeatureExtractor(grid_size=grid_size)
        
        # LeHDC classifier as a separate component
        self.lehdc = LeHDC(
            n_features=750,
            n_dimensions=n_dimensions,
            n_classes=n_classes,
            n_levels=n_levels,
            min_level=-1,
            max_level=1,
            epochs=30,
            dropout_rate=0.3,
            lr=0.01,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        self.lehdc_trained = False

    def forward(self, x):
        features = self.feature_network(x)
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

def test(name, model, testloader, device):
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
            # print(outputs)
            # print(outputs.shape)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted.to('cuda') == labels).sum().item()
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
    plt.savefig(f"./{name}_confusion_matrix_no_noise.png")

    print("Classification Report:")
    print(classification_report(all_labels, all_preds))
    with open(f"./{name}_classification_report_no_noise.txt", 'a', newline='') as file:
        file.truncate(0)
        file.write(f'Test Accuracy: {100 * correct / total:.2f}%, Test Loss: {test_loss}')
        file.write(classification_report(all_labels, all_preds))
    return

def test_with_noise(name, model, testloader, device, noise_std=0.1):
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
    plt.savefig(f"./{name}_confusion_matrix_noise_{noise_std}.png")
    plt.show()

    # Classification Report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds))
    with open(f"./{name}_classification_report_noise_{noise_std}.txt", 'a', newline='') as file:
        file.truncate(0)
        file.write(f'Test Accuracy: {accuracy:.2f}%, Test Loss: {test_loss}\n')
        file.write(classification_report(all_labels, all_preds))

    return accuracy, test_loss

def train_with_noisy_labels(model, train_loader, val_loader, learning_rate=0.001, epochs=10, 
                           noise_rate=0.2, device='cuda'):
    """
    Train a model with noisy labels.
    
    Args:
        model: The neural network model to train
        train_loader: DataLoader containing training data
        val_loader: DataLoader containing validation data
        learning_rate: Learning rate for optimizer
        epochs: Number of training epochs
        noise_rate: Percentage of labels to corrupt (0.0 to 1.0)
        device: Device to train on ('cuda' or 'cpu')
    
    Returns:
        Trained model and training history
    """
    model = model.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Track metrics
    train_losses = []
    val_losses = []
    val_accs = []
    best_acc = 0
    best_model = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Add noise to labels
            if noise_rate > 0:
                noisy_labels = add_label_noise(labels, noise_rate, num_classes=10)
                labels = noisy_labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model.classify(images)
            loss = loss_func(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
        
        # Calculate average training loss
        avg_train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        # Validate
        val_data = validate(model, val_loader, loss_func, device)
        val_acc = val_data[0]
        val_loss = val_data[1]
        val_accs.append(val_acc)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = copy.deepcopy(model)
    
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_acc': val_accs
    }
    
    return best_model, history

def add_label_noise(labels, noise_rate, num_classes=10):
    """
    Add noise to labels by randomly changing some labels to different classes.
    
    Args:
        labels: Tensor of true labels
        noise_rate: Percentage of labels to corrupt (0.0 to 1.0)
        num_classes: Number of classes in the dataset
    
    Returns:
        Tensor of noisy labels
    """
    noisy_labels = labels.clone()
    num_samples = len(labels)
    num_to_corrupt = int(noise_rate * num_samples)
    
    # Randomly select indices to corrupt
    indices_to_corrupt = torch.randperm(num_samples)[:num_to_corrupt]
    
    for idx in indices_to_corrupt:
        true_label = labels[idx].item()
        # Ensure the new label is different from the true label
        possible_labels = list(range(num_classes))
        possible_labels.remove(true_label)
        new_label = possible_labels[torch.randint(0, num_classes-1, (1,)).item()]
        
        noisy_labels[idx] = new_label
    
    return noisy_labels

def train(model, data, learning_rate, epochs, device, val_loader):
    accs = []
    val_losses = []
    losses = []
    best_acc = 0
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    batch_size = len(data)
    for epoch in tqdm.tqdm(range(epochs)):
        model.train()
        total_loss = 0
        for batch_index, (images, labels) in (enumerate(tqdm.tqdm(data, total=batch_size))):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model.classify(images)
            loss = loss_func(outputs, labels)
            total_loss += loss.item() * images.size(0)
            loss.backward()
            optimizer.step()
        losses.append(total_loss / len(data.dataset))
        val_data = validate(model, val_loader, loss_func, device)
        val_acc = val_data[0]
        accs.append(val_acc)
        val_losses.append(val_data[1])

        if val_acc > best_acc:
            best_acc = val_acc
            best_cnn = copy.deepcopy(model)

        print(f"Epoch {epoch+1}/{epochs}, Validation Accuracy: {val_acc:.2f}%")

    return best_cnn

def main(trainingmode=True, use_noisy_labels=True, noise_rate=0.2):
    num_epochs = 10
    batch_sz = 32
    learning_rate = 0.001
    name = "optimizedCNNLeHDC"
    model = KANCLeHDCModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    train_data = torchvision.datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
    other_data = torchvision.datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)
    val_data, test_data = torch.utils.data.random_split(other_data, [0.5, 0.5])

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_sz, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_sz, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_sz)
    
    if trainingmode:
        if use_noisy_labels:
            best_model, history = train_with_noisy_labels(
                model.feature_network, 
                trainloader, 
                valloader, 
                learning_rate=learning_rate, 
                epochs=num_epochs, 
                noise_rate=noise_rate, 
                device=device
            )
            
            # Plot training history
            # plt.figure(figsize=(12, 4))
            # plt.subplot(1, 2, 1)
            # plt.plot(history['train_loss'], label='Train Loss')
            # plt.plot(history['val_loss'], label='Val Loss')
            # plt.legend()
            # plt.title('Loss Curves with Noisy Labels')
            
            # plt.subplot(1, 2, 2)
            # plt.plot(history['val_acc'], label='Val Accuracy')
            # plt.legend()
            # plt.title('Validation Accuracy with Noisy Labels')
            # plt.savefig(f"./training_with_noisy_labels_{noise_rate}.png")
            
            torch.save(best_model.state_dict(), f"CNN_noisy_labels_{noise_rate}.pth")
            print(f"Model saved as CNN_noisy_labels_{noise_rate}.pth")
            test(name, best_model, testloader, device)
            test_with_noise(name, best_model, testloader, device, noise_std=0.1)
            test_with_noise(name, best_model, testloader, device, noise_std=0.4)
            test_with_noise(name, best_model, testloader, device, noise_std=0.7)
            test_with_noise(name, best_model, testloader, device, noise_std=1.0)
        else:
            best_model = train(model.feature_network, trainloader, learning_rate, num_epochs, device, valloader)
            torch.save(best_model.state_dict(), "CNN.pth")
            print("Model saved as CNN.pth")
    
    # Load the trained model
    if use_noisy_labels:
        model.feature_network.load_state_dict(torch.load(f"CNN_noisy_labels_{noise_rate}.pth", map_location=device))
    else:
        model.feature_network.load_state_dict(torch.load("CNN.pth", map_location=device))

    model.train_lehdc(trainloader, valloader)
    model.eval()

    # Test the model
    test(name, model, testloader, device)
    test_with_noise(name, model, testloader, device, noise_std=0.1)
    test_with_noise(name, model, testloader, device, noise_std=0.4)
    test_with_noise(name, model, testloader, device, noise_std=0.7)
    test_with_noise(name, model, testloader, device, noise_std=1.0)

if __name__ == '__main__':
    # Set to True to train with noisy labels, False for normal training
    use_noisy_labels = True
    noise_rate = 0.2  # Percentage of labels to corrupt
    main(trainingmode=True, use_noisy_labels=use_noisy_labels, noise_rate=noise_rate)
