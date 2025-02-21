import torch
import torch.nn as nn
import torchvision
import tqdm
import numpy as np
import copy
import matplotlib.pyplot as plt

class CNN_GP(nn.Module):
    def __init__(self):
        super(CNN_GP, self).__init__()
        self.conv_layer1 = nn.Conv2d(1, 3, 3)  # 28x28 -> 26x26
        self.max_pool1 = nn.MaxPool2d(2, 2)  # 26x26 -> 13x13
        self.conv_layer2 = nn.Conv2d(3, 6, 3)  # 13x13 -> 11x11
        self.conv_layer3 = nn.Conv2d(6, 2, 3)  # 11x11 -> 9x9

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # GMP instead of Flatten()
        
        self.fc1 = nn.Linear(2, 500)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(500, 10)  # FashionMNIST has 10 classes

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.max_pool1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)

        x = self.global_avg_pool(x)  # Output shape: (batch, 2, 1, 1)
        x = torch.flatten(x, 1)  # Flatten only last two dimensions

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

def train(model, data, learning_rate, epochs, device, val_loader):
    accs = []
    best_acc = None
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in tqdm.tqdm(range(epochs)):
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
        if best_acc is None or val_acc > best_acc:
            best_model = copy.deepcopy(model)
            best_acc = val_acc

        print(f"Epoch {epoch+1}/{epochs}, Validation Accuracy: {val_acc:.2f}%")
        epoch_axis = list(np.arange(epoch + 1) + 1)
        plt.figure()
        plt.plot(epoch_axis, accs)
        plt.xlabel("epoch")
        plt.ylabel("validation accuracy")
        plt.savefig("./cnn_global_pooling_val_acc")

    return best_model

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
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])

    train_data = torchvision.datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
    other_data = torchvision.datasets.FashionMNIST(root='data', train=True, download=False, transform=transform)
    val_data, test_data = torch.utils.data.random_split(other_data, [0.5, 0.5])

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_sz, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_sz, shuffle=True)

    model = CNN_GP().to(device)
    best_model = train(model, trainloader, learning_rate, epochs, device, valloader)

    # Save trained model
    torch.save(best_model.state_dict(), "models/cnn_global_pooling_fashion_MNIST.pth")
    print("Model saved as models/cnn_global_pooling_fashion_MNIST.pth")

if __name__ == '__main__':
    main()
