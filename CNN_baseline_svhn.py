import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import tqdm
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer1 = nn.Conv2d(3, 32, 3, padding=1)  
        self.max_pool1 = nn.MaxPool2d(2, 2)  
        self.conv_layer2 = nn.Conv2d(32, 64, 3, padding=1)  
        self.max_pool2 = nn.MaxPool2d(2, 2)  
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 256)  
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)  

    def forward(self, x):
        x = self.max_pool1(torch.relu(self.conv_layer1(x)))
        x = self.max_pool2(torch.relu(self.conv_layer2(x)))
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Training function
def train(model, data, learning_rate, epochs, device, val_loader):
    model.train()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_acc = 0
    best_model = None

    for epoch in tqdm.tqdm(range(epochs)):
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm.tqdm(data, total=len(data)):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        
        val_acc, val_loss = validate(model, val_loader, loss_func, device)
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = copy.deepcopy(model)

        print(f"Epoch {epoch+1}/{epochs}, Validation Accuracy: {val_acc:.2f}%, Loss: {total_loss/len(data.dataset):.4f}")

    return best_model

# Validation function
def validate(model, val_loader, loss_func, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss_func(outputs, labels).item() * images.size(0)

    return (100 * correct / total, total_loss / len(val_loader.dataset))

# Testing function
def test(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    loss_func = nn.CrossEntropyLoss()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss_func(outputs, labels).item() * images.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f'Test Accuracy: {100 * correct / total:.2f}%')
    print(f'Test Loss: {total_loss / len(testloader.dataset):.4f}')
    save_confusion_matrix(all_labels, all_preds, "cnn_svhn_confusion_matrix_no_noise.png")

# Test with noise function
def test_with_noise(model, testloader, device, noise_std=0.1):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    loss_func = nn.CrossEntropyLoss()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            
            noise = torch.randn_like(images) * noise_std
            noisy_images = images + noise
            noisy_images = torch.clamp(noisy_images, 0, 1)  

            outputs = model(noisy_images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss_func(outputs, labels).item() * images.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f'Test Accuracy with Noise (std={noise_std}): {100 * correct / total:.2f}%')
    print(f'Test Loss with Noise: {total_loss / len(testloader.dataset):.4f}')
    save_confusion_matrix(all_labels, all_preds, f"cnn_svhn_confusion_matrix_noise_{noise_std}.png")

# Function to save confusion matrix
def save_confusion_matrix(labels, preds, filename):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(filename)
    plt.show()

# Main function
def main(trainingmode=True):
    batch_size = 32
    epochs = 10
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))  
    ])


    train_data = torchvision.datasets.SVHN(root='data', split='train', download=True, transform=transform)
    test_data = torchvision.datasets.SVHN(root='data', split='test', download=True, transform=transform)

    # Split validation set
    num_val = len(test_data) // 2
    val_data, test_data = torch.utils.data.random_split(test_data, [num_val, len(test_data) - num_val])

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    model = CNN().to(device)

    if trainingmode:
        best_model = train(model, trainloader, learning_rate, epochs, device, valloader)
        torch.save(best_model.state_dict(), "models/cnn_baseline_svhn.pth")
        print("Model saved as models/cnn_baseline_svhn.pth")
    else:
        model.load_state_dict(torch.load("models/cnn_baseline_svhn.pth", map_location=device))

    test(model, testloader, device)
    for noise_std in [0.1, 0.4, 0.7, 1.0]:
        test_with_noise(model, testloader, device, noise_std)

if __name__ == '__main__':
    main()
