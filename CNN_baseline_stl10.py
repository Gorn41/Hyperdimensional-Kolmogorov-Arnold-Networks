import torch
import torch.nn as nn
import torchvision
import tqdm
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer1 = nn.Conv2d(3, 32, 3, padding=1)
        self.max_pool1 = nn.MaxPool2d(2, 2)
        self.conv_layer2 = nn.Conv2d(32, 64, 3, padding=1)
        self.max_pool2 = nn.MaxPool2d(2, 2)
        self.conv_layer3 = nn.Conv2d(64, 128, 3, padding=1)
        self.max_pool3 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 3 * 3, 256)  # Adjusted for STL-10 image size
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.max_pool1(torch.relu(self.conv_layer1(x)))
        x = self.max_pool2(torch.relu(self.conv_layer2(x)))
        x = self.max_pool3(torch.relu(self.conv_layer3(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

MODEL_PATH = "models/cnn_baseline_stl10.pth"

def train(model, data, learning_rate, epochs, device, val_loader):
    accs = []
    losses = []
    val_losses = []
    best_acc = 0
    best_cnn = None
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in tqdm.tqdm(data):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_func(outputs, labels)
            total_loss += loss.item() * images.size(0)
            loss.backward()
            optimizer.step()
        
        losses.append(total_loss / len(data.dataset))
        val_acc, val_loss = validate(model, val_loader, loss_func, device)
        accs.append(val_acc)
        val_losses.append(val_loss)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_cnn = copy.deepcopy(model)
        
        print(f"Epoch {epoch+1}/{epochs}, Validation Accuracy: {val_acc:.2f}%")
        
    return best_cnn

def validate(model, val_loader, loss_func, device):
    model.eval()
    correct, total, total_loss = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = loss_func(outputs, labels)
            total_loss += loss.item() * images.size(0)
    return 100 * correct / total, total_loss / len(val_loader.dataset)

def test(model, testloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Test Accuracy: {100 * correct / total:.2f}%')

def main(trainingmode=True):
    batch_sz = 64
    epochs = 15
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2241, 0.2215, 0.2239))
    ])


    train_data = torchvision.datasets.STL10(root='data', split='train', download=True, transform=transform)
    test_data = torchvision.datasets.STL10(root='data', split='test', download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_sz, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_sz)

    model = CNN().to(device)
    
    if trainingmode:
        best_model = train(model, trainloader, learning_rate, epochs, device, testloader)
        torch.save(best_model.state_dict(), MODEL_PATH)
        print(f"Model saved as {MODEL_PATH}")
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    test(model, testloader, device)

if __name__ == '__main__':
    main()
