import torch
import torch.nn as nn
import torchvision
import tqdm
import torch.nn.functional as F
from kan_convolutional.KANConv import KAN_Convolutional_Layer
import matplotlib.pyplot as plt
import copy
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import csv
import pandas as pd
import torchhd

class lehdcclassifier:

    def __init__(self, hvsize, roundingdp, numclasses, num_activations, batch_size=32):
        self.leHDC_clf = torchhd.classifiers.LeHDC(num_activations, hvsize, numclasses, device='cuda')
        self.roundingdp = roundingdp
        data = pd.read_csv("fashionmnist_KANCbaseline_activations.csv")
        # change this to include only all activations of target layer and activations of next layer/or true label
        data = data.loc[:, "linearlayer1_neuron_0":"true_label"]
        data = data.drop("linearlayer2_max_index", axis=1)
        # uncomment below to test subset of data for debugging
        data = data[:-59500]
        # data = data.round(roundingdp)
        data["true_label"] = data["true_label"].astype(int)
        features = torch.tensor(data.loc[:, "linearlayer1_neuron_0":"linearlayer1_neuron_199"].values, dtype=torch.float32)
        labels = torch.tensor(data["true_label"].values, dtype=torch.long)
        self.trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(features, labels), batch_size=batch_size, shuffle=False)

    def train_LeHDC(self):
        self.leHDC_clf.fit(self.trainloader)

    def logits_LeHDC(self, x):
        # x = torch.round(x, decimals=self.roundingdp)
        self.leHDC_clf.__call__(x)

    def feed_forward(self, x):
        # x = torch.round(x, decimals=self.roundingdp)
        print(x)
        self.leHDC_clf.forward(x)

    def predict_LeHDC(self, x):
        # x = torch.round(x, decimals=self.roundingdp)
        self.leHDC_clf.predict(x)

class KANC_HDC(nn.Module):
    def __init__(self,grid_size: int = 5):
        super().__init__()
        self.conv1 = KAN_Convolutional_Layer(in_channels=1,
            out_channels= 5,
            kernel_size= (3,3),
            grid_size = grid_size
        )

        self.conv2 = KAN_Convolutional_Layer(in_channels=5,
            out_channels= 5,
            kernel_size = (5,5),
            grid_size = grid_size
        )

        self.conv3 = KAN_Convolutional_Layer(in_channels=5,
            out_channels= 2,
            kernel_size = (3,3),
            grid_size = grid_size
        )

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flat = nn.Flatten()
        self.linearlayer1 = nn.Linear(98, 200)
        self.relu = nn.ReLU()
        self.linearlayer2 = nn.Linear(200, 10)
        self.name = f"KANC MLP (Small) (gs = {grid_size})"
        self.hdc_clf = lehdcclassifier(10000, 1, 10, 200)

    
    def trainhdclayer(self):
        self.hdc_clf.train_LeHDC()


    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flat(x)
        x = self.relu(self.linearlayer1(x))
        x = self.hdc_clf.feed_forward(x)
        return x
    
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

def main():
    batch_sz = 32
    # epochs = 10
    # learning_rate = 0.001

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    train_data = torchvision.datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
    other_data = torchvision.datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)
    val_data, test_data = torch.utils.data.random_split(other_data, [0.5, 0.5])

    # trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_sz, shuffle=True)
    # valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_sz, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_sz)

    # load in the original KANC-MLP model weights but make the forward function realy one those, but at the end it should be sent to the HDC linear layer 2?
    model = KANC_HDC().to(device)

    model.load_state_dict(torch.load("models/KANC_MLP.pth", map_location=device))
    model.trainhdclayer()
    # model = KANC_HDC().to(device)

    # # test saved model with noise
    # model.load_state_dict(torch.load("models/KANC_MLP.pth", map_location=device))


    test(model, testloader, device)
    test_with_noise(model, testloader, device, noise_std=0.1)
    test_with_noise(model, testloader, device, noise_std=0.4)
    test_with_noise(model, testloader, device, noise_std=0.7)
    test_with_noise(model, testloader, device, noise_std=1.0)

    

if __name__ == '__main__':
    main()