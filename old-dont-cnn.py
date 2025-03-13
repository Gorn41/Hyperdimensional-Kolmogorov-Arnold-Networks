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
    def __init__(self, n_dimensions=10000, n_classes=10, n_levels=100, grid_size=5):
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
            epochs=10,
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
    plt.title('HDC-reinforced-KANC Confusion Matrix (No Noise)')
    plt.savefig("./HDC-reinforced-KANC_fashionmnist_confusion_matrix_no_noise.png")
    plt.show()

    print("Classification Report:")
    print(classification_report(all_labels, all_preds))
    with open("./HDC-reinforced-KANC_classification_report_no_noise.txt", 'a', newline='') as file:
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
    plt.title(f'HDC-reinforced-KANC Confusion Matrix (Noise Std = {noise_std})')
    plt.savefig(f"./HDC-reinforced-KANC_fashionmnist_confusion_matrix_noise_{noise_std}.png")
    plt.show()

    # Classification Report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds))
    with open(f"./HDC-reinforced-KANC_classification_report_noise_{noise_std}.txt", 'a', newline='') as file:
        # clear file
        file.truncate(0)
        file.write(f'Test Accuracy: {accuracy:.2f}%, Test Loss: {test_loss}\n')
        file.write(classification_report(all_labels, all_preds))

    return accuracy, test_loss

def train(model, data, learning_rate, epochs, device, val_loader):
    accs = []
    val_losses = []
    losses = []
    best_acc = 0
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
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

        plt.figure()
        plt.plot(np.arange(1, epoch + 2), accs)
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy")
        plt.title("KANC Baseline Validatation Accuracy over Epochs")
        plt.savefig("./kanc_baseline_fashionmnist_val_acc.png")
        plt.figure()
        plt.plot(np.arange(1, epoch + 2), losses)
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title("KANC Baseline Training Loss over Epochs")
        plt.savefig("./kanc_baseline_fashionmnist_training_loss.png")
        plt.figure()
        plt.plot(np.arange(1, epoch + 2), val_losses)
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.title("KANC Baseline Validatation Loss over Epochs")
        plt.savefig("./kanc_baseline_fashionmnist_val_loss.png")
        plt.close('all')
    return best_cnn

def main():
    num_epochs = 5
    batch_sz = 32

    model = KANCLeHDCModel()

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    # train_dataset = MNIST("./data", train=True, transform=transform, download=True)


    # dataset_size = len(train_dataset)
    # train_subset = Subset(train_dataset, range(dataset_size // 50))

    # train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    # test_dataset = MNIST("./data", train=False, transform=transform, download=True)

    # dataset_size = len(test_dataset)
    # test_subset = Subset(test_dataset, range(dataset_size // 50))
    # test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)


    train_data = torchvision.datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
    other_data = torchvision.datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)
    val_data, test_data = torch.utils.data.random_split(other_data, [0.5, 0.5])

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_sz, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_sz, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_sz)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    best_model = train(model.feature_network, trainloader, 0.001, 10, device, valloader)
    torch.save(best_model.state_dict(), "KANC_MLP.pth")
    print("Model saved as KANC_MLP.pth")

    model.feature_network.load_state_dict(torch.load("KANC_MLP.pth", map_location=device))

    model.train_lehdc(trainloader, valloader)

    model.eval()

    test(model, testloader, device)
    test_with_noise(model, testloader, device, noise_std=0.1)
    test_with_noise(model, testloader, device, noise_std=0.4)
    test_with_noise(model, testloader, device, noise_std=0.7)
    test_with_noise(model, testloader, device, noise_std=1.0)

if __name__ == '__main__':
    main()