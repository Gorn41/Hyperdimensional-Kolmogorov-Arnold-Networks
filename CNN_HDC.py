import torch
import torch.nn as nn
import torchvision
import tqdm
import numpy as np
import copy
import matplotlib.pyplot as plt
from HDC import hdc_model

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer1 = nn.Conv2d(1, 3, 3)  # Input: (1, 28, 28) -> Output: (3, 26, 26)
        self.max_pool1 = nn.MaxPool2d(2, 2)  # Output: (3, 13, 13)
        self.conv_layer2 = nn.Conv2d(3, 6, 3)  # Output: (6, 11, 11)
        self.conv_layer3 = nn.Conv2d(6, 2, 3)  # Output: (2, 9, 9)

        self.hdc_modules = hdc_model(1000, 9)

        self.fc1 = nn.Linear(2 * 1000, 20000)  # Adjusted to match output size
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20000, 10)  # 10 classes for FashionMNIST

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.max_pool1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        
        x = self.hdc_modules.hdc_flatten_concat(x)
        x = x.to("cuda")

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train(model, data, learning_rate, epochs, device, val_loader):
    accs = []
    best_acc = 0
    best_cnn = None
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in tqdm.tqdm(range(epochs)):
        model.train()
        for batch_index, (images, labels) in enumerate(data):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

        val_acc = validate(model, val_loader, device)
        accs.append(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_cnn = copy.deepcopy(model)

        print(f"Epoch {epoch+1}/{epochs}, Validation Accuracy: {val_acc:.2f}%")
        
        plt.figure()
        plt.plot(np.arange(1, epoch + 2), accs)
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy")
        plt.savefig("./cnn_fashionmnist_val_acc.png")
    
    return best_cnn

def validate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def main():
    batch_sz = 32
    epochs = 10
    learning_rate = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = torchvision.datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
    other_data = torchvision.datasets.FashionMNIST(root='data', train=True, download=False, transform=transform)
    val_data, test_data = torch.utils.data.random_split(other_data, [0.5, 0.5])

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_sz, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_sz, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_sz)

    model = CNN().to(device)
    best_model = train(model, trainloader, learning_rate, epochs, device, valloader)
    
    torch.save(best_model.state_dict(), "models/cnn_baseline_fashion_MNIST.pth")
    print("Model saved as models/cnn_baseline_fashion_MNIST.pth")

if __name__ == '__main__':
    main()

