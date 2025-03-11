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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class hdc_linear_layer2:
    def __init__(self, hvsize, roundingdp, numclasses, num_activations, quantization_bins=16):
        self.hvsize = hvsize
        self.numclasses = numclasses
        self.class_hvs = torch.empty((self.numclasses, self.hvsize), dtype=torch.int).to('cuda')
        self.roundingdp = roundingdp
        self.num_activations = num_activations
        self.quantization_bins = quantization_bins
        self.codebook = {}
        
        # Load activations
        data = pd.read_csv("fashionmnist_KANCbaseline_activations.csv")
        training_data = data.loc[:, "linearlayer1_neuron_0":"true_label"].drop("linearlayer2_max_index", axis=1)
        training_data = training_data[:-59500]
        
        # Store training groups
        self.training_groups = {}
        for i in range(self.numclasses):
            self.training_groups[i] = self.quantize(torch.tensor(training_data[training_data["true_label"] == int(i)].iloc[:, :-1].to_numpy()).to('cuda'))
    
    def quantize(self, tensor):
        """Quantizes activations into discrete bins"""
        min_val, max_val = tensor.min(), tensor.max()
        bins = torch.linspace(min_val, max_val, self.quantization_bins).to(tensor.device)
        return torch.bucketize(tensor, bins) - 1  # Shift to 0-based index

    def set(self, symbol):
        """Assigns a random hypervector to each unique activation level"""
        self.codebook[symbol] = torchhd.BSCTensor.random(1, self.hvsize).to('cuda')

    def get(self, symbol):
        return self.codebook[symbol]
    
    def process_row(self, row):
        """Maps activations to hypervectors and binds them"""
        res = torch.empty((row.shape[0], self.hvsize), dtype=torch.float).to('cuda')
        for i, value in enumerate(row):
            if value.item() not in self.codebook:
                self.set(value.item())
            res[i] = self.get(value.item())
        return torchhd.bind(res)  # Bind all activation hypervectors

    def trainhdc(self):
        """Trains HDC layer by encoding activations into hypervectors and computing class representations"""
        for i in range(self.numclasses):
            examples = self.training_groups[i]
            res = torch.stack([self.process_row(example) for example in examples])
            self.class_hvs[i] = torchhd.bundle(res).int()

    def hdc_forward(self, x):
        x = self.quantize(torch.round(x, decimals=self.roundingdp))
        input_hvs = torch.stack([self.process_row(sample) for sample in x])
        return torchhd.hamming_similarity(input_hvs, self.class_hvs)
    
    def visualize_hypervectors(self, use_tsne=False):
        """Visualizes activation hypervectors in latent space"""
        activations, labels = [], []
        for class_id, hv_group in self.training_groups.items():
            for hv in hv_group:
                activations.append(self.process_row(hv).cpu().numpy())
                labels.append(class_id)
        
        activations = torch.tensor(activations)
        reducer = TSNE(n_components=2) if use_tsne else PCA(n_components=2)
        activations_2d = reducer.fit_transform(activations)
        
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(activations_2d[:, 0], activations_2d[:, 1], c=labels, cmap="tab10", alpha=0.7)
        plt.colorbar(scatter, label="Class Label")
        plt.xlabel("Component 1" if not use_tsne else "t-SNE Dim 1")
        plt.ylabel("Component 2" if not use_tsne else "t-SNE Dim 2")
        plt.title(f"Hypervector Latent Space Visualization ({'t-SNE' if use_tsne else 'PCA'})")
        plt.show()

class KANC_HDC(nn.Module):
    def __init__(self, grid_size: int = 5):
        super().__init__()
        self.conv1 = KAN_Convolutional_Layer(in_channels=1, out_channels=5, kernel_size=(3,3), grid_size=grid_size)
        self.conv2 = KAN_Convolutional_Layer(in_channels=5, out_channels=5, kernel_size=(5,5), grid_size=grid_size)
        self.conv3 = KAN_Convolutional_Layer(in_channels=5, out_channels=2, kernel_size=(3,3), grid_size=grid_size)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flat = nn.Flatten()
        self.linearlayer1 = nn.Linear(98, 200)
        self.relu = nn.ReLU()
        self.linearlayer2 = nn.Linear(200, 10)
        self.hdc_clf = hdc_linear_layer2(10000, 2, 10, 200)

    def trainhdclayer(self):
        self.hdc_clf.trainhdc()
        self.hdc_clf.visualize_hypervectors(use_tsne=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flat(x)
        x = self.relu(self.linearlayer1(x))
        x = self.hdc_clf.hdc_forward(x)
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
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted.to('cuda') == labels).sum().item()
            loss = loss_func(outputs.float(), labels)
            total_loss += loss.item() * images.size(0)
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

    # visualize model
    model.visualize_hypervectors(use_tsne=True)
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