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

class hdc_linear_layer2:
    def __init__(self, hvsize, roundingdp, numclasses):
        self.hvsize = hvsize
        self.codebook = {}
        self.numclasses = numclasses
        self.class_hvs = torch.empty((self.numclasses, self.hvsize), dtype=torch.bool)
        self.roundingdp = roundingdp
        data = pd.read_csv("fashionmnist_KANCbaseline_activations.csv")
        training_data = data.loc[:, "linearlayer1_neuron_0":"linearlayer2_max_index"]
        self.training_groups = {}
        for i in range(self.numclasses):
            self.training_groups[i] = np.round(training_data[training_data["linearlayer2_max_index"] == i].iloc[:, :-1], self.roundingdp)

    def set(self, symbol):
        self.codebook[symbol] = torchhd.BSCTensor.random(1, self.hvsize, dtype=torch.long)
        return

    def get(self, symbol):
        return self.codebook[symbol]
    
    def process_row(self, row):
        n_features = row.shape[0]
        res = torch.empty((n_features, self.hvsize), dtype=torch.bool)
        for i in range(n_features):
            value = row.iloc[i]
            if value not in self.codebook:
                    self.set(value)
            value_hv = self.get(value)
            for k in range(i):
                value_hv = torchhd.permute(value_hv)
            res[i] = value_hv
        return torchhd.multibundle(res)

    def train(self):
        for i in tqdm.tqdm(range(self.numclasses)):
            n_examples = self.training_groups[i].shape[0]
            res = torch.empty((n_examples, self.hvsize), dtype=torch.bool)
            res_idx = 0
            for _, row in tqdm.tqdm(self.training_groups[i].iterrows()):
                res[res_idx] = self.process_row(row)
                res_idx += 1
            self.class_hvs[i] = torchhd.multibundle(res)
        return

    def hdc_forward(self, x):
        x = np.round(pd.DataFrame(x), self.roundingdp)
        res = torch.empty((x.shape[0], self.numclasses), dtype=torch.int)
        res_idx = 0
        for _, row in x.iterrows():
            activation_hv = self.process_row(row)
            # if error convert MAPTensor to normal tensor
            res[res_idx] = torchhd.hamming_similarity(activation_hv, self.class_hvs)
            res_idx += 1
        return res
    
    def save_codebook(self):
        with open("HDC-reinforced-KANC-codebook-FC2.csv", "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.codebook.keys())
            writer.writerow(self.codebook.values())
        return
    
    def save_class_hvs(self):
        pd.DataFrame(self.class_hvs).to_csv('HDC-reinforced-KANC-Class-HVs-FC2.csv', index=False, header=False)
        return
    
    def save_model(self):
        self.save_codebook()
        self.save_class_hvs()
        return


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

        self.conv3 = KAN_Convolutional_Layer(in_channels=5,
            out_channels= 2,
            kernel_size = (3,3),
            grid_size = grid_size
        )

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flat = nn.Flatten()
        self.linearlayer1 = nn.Linear(162, 500)
        self.relu = nn.ReLU()
        self.linearlayer2 = nn.Linear(500, 10)
        self.name = f"KANC MLP (Small) (gs = {grid_size})"
        self.hdc_clf = hdc_linear_layer2(10000, 2, 10)
        self.hdc_clf.train()
        self.hdc_clf.save_model()


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
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = loss_func(outputs, labels)
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

def main(trainingmode=True):

    batch_sz = 32
    epochs = 10
    learning_rate = 0.001

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    train_data = torchvision.datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
    other_data = torchvision.datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)
    val_data, test_data = torch.utils.data.random_split(other_data, [0.5, 0.5])

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_sz, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_sz, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_sz)

    model = KANC_MLP().to(device)

    if trainingmode:
        best_model = train(model, trainloader, learning_rate, epochs, device, valloader)
        test(best_model, testloader, device)

        torch.save(model.state_dict(), "models/KANC_MLP.pth")
        print("Model saved as models/KANC_MLP.pth")

    # test saved model with noise
    model.load_state_dict(torch.load("models/KANC_MLP.pth", map_location=device))
    test(model, testloader, device)
    test_with_noise(model, testloader, device, noise_std=0.1)
    test_with_noise(model, testloader, device, noise_std=0.4)
    test_with_noise(model, testloader, device, noise_std=0.7)
    test_with_noise(model, testloader, device, noise_std=1.0)

    

if __name__ == '__main__':
    main()