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
    def __init__(self, hvsize, roundingdp, numclasses, num_activations):
        self.hvsize = hvsize
        self.codebook = {}
        self.numclasses = numclasses
        self.class_hvs = torch.empty((self.numclasses, self.hvsize), dtype=torch.int)
        self.roundingdp = roundingdp
        self.num_activations = num_activations
        data = pd.read_csv("fashionmnist_KANCbaseline_activations.csv")
        # change this to include only all activations of target layer and activations of next layer/or true label
        training_data = data.loc[:, "linearlayer1_neuron_0":"true_label"]
        training_data = training_data.drop("linearlayer2_max_index", axis=1)
        # uncomment below to test subset of data for debugging
        training_data = training_data[:-59500]
        self.training_groups = {}
        for i in range(self.numclasses):
            self.training_groups[i] = np.round(training_data[training_data["true_label"] == i].iloc[:, :-1], self.roundingdp)
        

    def set(self, symbol):
        self.codebook[symbol] = torchhd.BSCTensor.random(1, self.hvsize)
        return

    def get(self, symbol):
        return self.codebook[symbol]
    
    def process_row(self, row):
        n_features = row.shape[0]
        res = torch.empty((n_features, self.hvsize), dtype=torch.float)
        for i in range(n_features):
            value = row.iloc[i]
            if value not in self.codebook:
                    self.set(value)
            res[i] = self.get(value)
        return res

    def train(self):
        for i in tqdm.tqdm(range(self.numclasses)):
            n_examples = self.training_groups[i].shape[0]
            res = torch.empty((n_examples, self.num_activations, self.hvsize), dtype=torch.float)
            res_idx = 0
            for _, row in tqdm.tqdm(self.training_groups[i].iterrows()):
                res[res_idx] = self.process_row(row)
                res_idx += 1
            for j in self.num_activations:
                res[:, j, :] = torchhd.permute(res[:, j, :], shifts=j)
            res = torch.mean(res, axis=1)
            rounded_tensor = torch.round(res)
            half_values = (res == 0.5)
            random_choices = torch.bernoulli(torch.ones_like(res[half_values], dtype=torch.float32) * 0.5).int()
            rounded_tensor[half_values] = random_choices
            self.class_hvs[i] = torchhd.multibundle(rounded_tensor)
        return

    def hdc_forward(self, x):
        x = np.round(pd.DataFrame(x), self.roundingdp)
        # res = torch.empty((x.shape[0], self.numclasses), dtype=torch.int)
        inputhvs = torch.empty((x.shape[0], self.num_activations, self.hvsize), dtype=torch.float)
        input_idx = 0
        for _, row in x.iterrows():
            inputhvs[input_idx] = self.process_row(row)
            # if error convert MAPTensor to normal tensor
            input_idx += 1
        for j in self.num_activations:
            inputhvs[:, j, :] = torchhd.permute(inputhvs[:, j, :], shifts=j)
        inputhvs = torch.mean(inputhvs, axis=1)
        rounded_tensor = torch.round(inputhvs)
        half_values = (inputhvs == 0.5)
        random_choices = torch.bernoulli(torch.ones_like(inputhvs[half_values], dtype=torch.float) * 0.5).int()
        rounded_tensor[half_values] = random_choices
        res = torchhd.hamming_similarity(rounded_tensor.int(), self.class_hvs)
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
        self.hdc_clf = hdc_linear_layer2(10000, 2, 10, 200)
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