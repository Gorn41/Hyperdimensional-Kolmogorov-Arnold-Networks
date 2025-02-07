import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        self.conv_layer1 = nn.Conv2d(3, 5, 5)
        self.max_pool1 = nn.MaxPool2d(2, 2)
        self.conv_layer2 = nn.Conv2d(5, 10, 5)
        self.max_pool2 = nn.MaxPool2d(2, 2)
        self.conv_layer3 = nn.Conv2d(10, 12, 3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1452, 3000)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3000, 200)

    def forward_pass(self, x):
    
        x = self.conv_layer1(x)
        x = self.max_pool1(x)
        x = self.conv_layer2(x)
        x = self.max_pool2(x)
        x = self.conv_layer3(x)

        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

