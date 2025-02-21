import torch
import torch.nn as nn
import torchvision
import tqdm
import numpy as np

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer1 = nn.Conv2d(1, 3, 3)
        self.max_pool1 = nn.MaxPool2d(2, 2)
        self.conv_layer2 = nn.Conv2d(3, 6, 3)
        self.conv_layer3 = nn.Conv2d(6, 2, 3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(162, 500)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(500, 10)

    def forward_pass(self, x):
    
        x = self.conv_layer1(x)
        x = self.max_pool1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        print(type(x))
        print(x.shape)
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


def train(model, data, learning_rate, epochs, device, val_loader):
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in tqdm.tqdm(range(epochs)):
        for batch_index, (images, labels) in enumerate(data):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward_pass(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
        val_acc = validate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs}, Validation Accuracy: {val_acc:.2f}%")
    return

def validate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model.forward_pass(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def test():
    return


def main():

    batch_sz = 32
    epochs = 10

    learning_rate = 0.001

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")


    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    train_data = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform)
    other_data = torchvision.datasets.MNIST(root='data', train=True, download=False, transform=transform)
    val_data, test_data = torch.utils.data.random_split(other_data, [0.5, 0.5])
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_sz, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_sz, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_sz)

    model = CNN()
    model = model.to(device)
    train(model, trainloader, learning_rate, epochs, device, valloader)

    return

if __name__ == '__main__':
    main()