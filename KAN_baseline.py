import torch
import torch.nn as nn
import torchvision
import tqdm
import torch.nn.functional as F
from kan_convolutional.KANConv import KAN_Convolutional_Layer
import matplotlib.pyplot as plt
import copy
import numpy as np


class KANC_MLP(nn.Module):
    def __init__(self,grid_size: int = 5):
        super().__init__()
        self.conv1 = KAN_Convolutional_Layer(in_channels=1,
            out_channels= 5,
            kernel_size= (3,3),
            grid_size = grid_size
        )

        self.conv2 = KAN_Convolutional_Layer(in_channels=5,
            out_channels= 5,
            kernel_size = (3,3),
            grid_size = grid_size
        )

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flat = nn.Flatten()
        self.linear1 = nn.Linear(125, 10)
        self.name = f"KANC MLP (Small) (gs = {grid_size})"


    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.flat(x)
        x = self.linear1(x)
        x = F.log_softmax(x, dim=1)
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
            outputs = model.forward(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
        val_acc = validate(model, val_loader, device)
        accs.append(val_acc)
        if best_acc == None or val_acc > best_acc:
            best_cnn = copy.deepcopy(model)
        print(f"Epoch {epoch+1}/{epochs}, Validation Accuracy: {val_acc:.2f}%")
        epoch_axis = list(np.arange(epoch + 1) + 1)
        plt.figure()
        plt.plot(epoch_axis, accs)
        plt.xlabel("epoch")
        plt.ylabel("validation accuracy")
        plt.savefig("./cnn_baseline_val_err")

def validate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model.forward(images)
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

    model = KANC_MLP()
    model = model.to(device)
    train(model, trainloader, learning_rate, epochs, device, valloader)

    return

if __name__ == '__main__':
    main()
