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


    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flat(x)
        x = self.relu(self.linearlayer1(x))
        x = self.linearlayer2(x)
        return x

def train(model, data, learning_rate, epochs, device, val_loader):
    accs = []
    val_losses = []
    losses = []
    best_acc = 0
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    batch_size = len(data)
    for epoch in tqdm.tqdm(range(epochs)):
        total_loss = 0
        for batch_index, (images, labels) in (enumerate(tqdm.tqdm(data, total=batch_size))):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(images)
            loss = loss_func(outputs, labels)
            total_loss += loss.item() * images.size(0)
            loss.backward()
            optimizer.step()
        losses.append(total_loss / len(data.dataset))
        val_data = validate(model, val_loader, loss_func, device)
        val_acc = val_data[0]
        accs.append(val_acc)
        val_losses.append(val_data[1])

        if val_acc > best_acc:
            best_acc = val_acc
            best_cnn = copy.deepcopy(model)

        print(f"Epoch {epoch+1}/{epochs}, Validation Accuracy: {val_acc:.2f}%")

        plt.figure()
        plt.plot(np.arange(1, epoch + 2), accs)
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy")
        plt.title("KANC Baseline Validatation Accuracy over Epochs")
        plt.savefig("./kanc_baseline_fashionmnist_val_acc.png")
        plt.figure()
        plt.plot(np.arange(1, epoch + 2), losses)
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title("KANC Baseline Training Loss over Epochs")
        plt.savefig("./kanc_baseline_fashionmnist_training_loss.png")
        plt.figure()
        plt.plot(np.arange(1, epoch + 2), val_losses)
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.title("KANC Baseline Validatation Loss over Epochs")
        plt.savefig("./kanc_baseline_fashionmnist_val_loss.png")
        plt.close('all')
    return best_cnn

def validate(model, val_loader, loss_func, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model.forward(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = loss_func(outputs, labels)
            total_loss += loss.item() * images.size(0)
    return (100 * correct / total, total_loss / len(val_loader.dataset))

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
    plt.title('KANC Baseline Confusion Matrix (No Noise)')
    plt.savefig("./kanc_baseline_fashionmnist_confusion_matrix_no_noise.png")
    plt.show()

    print("Classification Report:")
    print(classification_report(all_labels, all_preds))
    with open("./KANC_baseline_classification_report_no_noise.txt", 'a', newline='') as file:
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
    plt.title(f'KANC Baseline Confusion Matrix (Noise Std = {noise_std})')
    plt.savefig(f"./KANC_baseline_fashionmnist_confusion_matrix_noise_{noise_std}.png")
    plt.show()

    # Classification Report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds))
    with open(f"./KANC_baseline_classification_report_noise_{noise_std}.txt", 'a', newline='') as file:
        # clear file
        file.truncate(0)
        file.write(f'Test Accuracy: {accuracy:.2f}%, Test Loss: {test_loss}\n')
        file.write(classification_report(all_labels, all_preds))

    return accuracy, test_loss

def main(trainingmode=False):

    batch_sz = 32
    epochs = 10
    learning_rate = 0.001

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    train_data = torchvision.datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
    other_data = torchvision.datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)
    dataset_size = len(other_data)
    split1 = dataset_size // 2 
    split2 = dataset_size - split1 
    val_data, test_data = torch.utils.data.random_split(other_data, [split1, split2])

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_sz, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_sz, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_sz)

    model = KANC_MLP().to(device)
    print(model.state_dict().keys())


    if trainingmode:
        best_model = train(model, trainloader, learning_rate, epochs, device, valloader)
        test(best_model, testloader, device)

        torch.save(model.state_dict(), "models/KANC_MLP.pth")
        print("Model saved as models/KANC_MLP.pth")

    # test saved model with noise
    model_state_dict = torch.load("models/KANC_MLP.pth", map_location=device)
    print(model_state_dict.keys())
    # model.load_state_dict(torch.load("models/KANC_MLP.pth", map_location=device))
    # print(model.keys())

    test(model, testloader, device)
    test_with_noise(model, testloader, device, noise_std=0.1)
    test_with_noise(model, testloader, device, noise_std=0.4)
    test_with_noise(model, testloader, device, noise_std=0.7)
    test_with_noise(model, testloader, device, noise_std=1.0)

    model.eval()
    activations = {}
    hooks = []
    def capture_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach().cpu.numpy()
        return hook

    for name, layer in model.named_children():
        # if you want all conv layers change conv3 below to just conv
        if name.startswith('conv3') or name.startswith('linearlayer'):
            hooks.append(layer.register_forward_hook(capture_activation(name)))

    headers = []
    for name, layer in model.named_children():
        if name.startswith('conv'):
            # if name == 'conv1':
            #     n_features = 26 * 26 * 5
            #     headers += [f"{name}_neuron_{i}" for i in range(n_features)]
            # if name == 'conv2':
            #     n_features = 11 * 11 * 5
            #     headers += [f"{name}_neuron_{i}" for i in range(n_features)]
            if name == 'conv3':
                n_features = 9 * 9 * 2
                headers.extend([f"{name}_neuron_{i}" for i in range(n_features)])
        if name.startswith('linearlayer'):
            if name == 'linearlayer2':
                headers.append(f"{name}_max_index")  # Single header for max index
            else:
                headers.extend([f"{name}_neuron_{i}" for i in range(layer.out_features)])

    headers.append("true label")
        
    with open('fashionmnist_KANCbaseline_activations.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)
            print(type(labels))
            print(labels.shape)
            with torch.no_grad():
                model.forward(images)

            batch_size = images.size(0)
            for i in range(batch_size):
                row = []
                # row.extend(activations['conv1'][i].flatten())
                # row.extend(activations['conv2'][i].flatten())
                row.extend(activations['conv3'][i].flatten())
                row.extend(activations['linearlayer1'][i].flatten())
                linearlayer2_output = activations['linearlayer2'][i]
                max_idx = linearlayer2_output.argmax()
                row.append(max_idx)
                row.append(labels[i].item()) # True label
                writer.writerow(row)

            activations.clear() 

    for hook in hooks:
        hook.remove()
    return

if __name__ == '__main__':
    main()