import torch
import torch.nn as nn
import torchvision
import tqdm
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer1 = nn.Conv2d(1, 5, 3)  # Input: (1, 28, 28) -> Output: (3, 26, 26)
        self.max_pool1 = nn.MaxPool2d(2, 2)  # Output: (3, 13, 13)
        self.conv_layer2 = nn.Conv2d(5, 5, 3)  # Output: (6, 11, 11)
        self.conv_layer3 = nn.Conv2d(5, 2, 3)  # Output: (6, 11, 11)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(162, 500)  # Adjusted to match output size
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(500, 10)  # 10 classes for FashionMNIST

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.max_pool1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train(model, data, learning_rate, epochs, device, val_loader):
    accs = []
    losses = []
    val_losses = []
    best_acc = 0
    best_cnn = None
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
        plt.title("CNN baseline Validatation Accuracy over Epochs")
        plt.savefig("./cnn_fashionmnist_val_acc.png")
        plt.figure()
        plt.plot(np.arange(1, epoch + 2), losses)
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title("CNN baseline Training Loss over Epochs")
        plt.savefig("./cnn_fashionmnist_training_loss.png")
        plt.figure()
        plt.plot(np.arange(1, epoch + 2), val_losses)
        plt.xlabel("Epoch")
        plt.title("CNN baseline Validatation Loss over Epochs")
        plt.ylabel("Validation Loss")
        plt.savefig("./cnn_fashionmnist_val_loss.png")
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
    plt.title('CNN Baseline Confusion Matrix (No Noise)')
    plt.savefig("./cnn_fashionmnist_confusion_matrix_no_noise.png")
    plt.show()

    print("Classification Report:")
    print(classification_report(all_labels, all_preds))
    with open("./cnn_baseline_classification_report_no_noise.txt", 'a', newline='') as file:
        file.write(f'Test Accuracy: {100 * correct / total:.2f}%, Test Loss: {test_loss}')
        file.write(classification_report(all_labels, all_preds)) 

    print("done")
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
    plt.title(f'CNN Baseline Confusion Matrix (Noise Std = {noise_std})')
    plt.savefig(f"./cnn_fashionmnist_confusion_matrix_noise_{noise_std}.png")
    plt.show()

    # Classification Report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds))
    with open(f"./cnn_baseline_classification_report_noise_{noise_std}.txt", 'a', newline='') as file:
        file.write(f'Test Accuracy: {accuracy:.2f}%, Test Loss: {test_loss}\n')
        file.write(classification_report(all_labels, all_preds)) 

    return accuracy, test_loss


def main(trainingmode=True):
    batch_sz = 32
    epochs = 10
    learning_rate = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = torchvision.datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
    other_data = torchvision.datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)
    val_data, test_data = torch.utils.data.random_split(other_data, [0.5, 0.5])

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_sz, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_sz, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_sz)

    # train model
    model = CNN().to(device)

    if trainingmode:
        best_model = train(model, trainloader, learning_rate, epochs, device, valloader)
        test(best_model, testloader, device)
        
        torch.save(best_model.state_dict(), "models/cnn_baseline_fashion_MNIST.pth")
        print("Model saved as models/cnn_baseline_fashion_MNIST.pth")

    # test saved model with noise
    model.load_state_dict(torch.load("models/cnn_baseline_fashion_MNIST.pth", map_location=torch.device('cuda')))
    test(model, testloader, device)
    test_with_noise(model, testloader, device, noise_std=0.1)
    test_with_noise(model, testloader, device, noise_std=0.4)
    test_with_noise(model, testloader, device, noise_std=0.7)
    test_with_noise(model, testloader, device, noise_std=1.0)

if __name__ == '__main__':
    main()

