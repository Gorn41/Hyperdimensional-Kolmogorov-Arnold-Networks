import torch
import torch.nn as nn
import torch.nn.functional as F
import torchhd
from torchhd.classifiers import LeHDC
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision

class LeHDCCNN(nn.Module):
    def __init__(self, hdc_dimensions=1000, n_classes=10, dropout_rate=0.1, n_levels=200):
        super(LeHDCCNN, self).__init__()
        

        self.conv1 = nn.Conv2d(1, 5, kernel_size=3)
        self.conv2 = nn.Conv2d(5, 5, kernel_size=3)
        self.conv3 = nn.Conv2d(5, 4, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        self.fc = nn.Linear(324, 1024)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout_rate)

        self.lehdc = LeHDC(
            n_features=1024,           
            n_dimensions=hdc_dimensions,
            n_classes=n_classes,
            n_levels=n_classes,             # you can kind of think of this as rounding
            min_level=-1,             # don't change this
            max_level=1,              # don'change this
            epochs=10,                # scale with the other epoch param
            lr=0.0000001              # don't change this
        )

        
    def forward(self, x):
        # Feature extraction
        x = self.pool(F.relu(self.conv1(x)))
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Flatten
        x = self.flatten(x)
        
        # Linear layer with tanh activation
        x = self.fc(x)
        x = self.dropout(x)
        # LeHDC classification
        x = self.lehdc(x)
        
        return x

# Load MNIST dataset
def load_mnist_data(batch_size=34):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_data = torchvision.datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
    other_data = torchvision.datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)
    val_data, test_data = torch.utils.data.random_split(other_data, [0.5, 0.5])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    
    return train_loader, valloader, test_loader

# Training function
def train(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}, Accuracy: {100.*correct/total:.2f}%')
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

# Testing function
def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
    return test_loss, test_acc

# Main function to run training and testing
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10
    hdc_dimensions = 1024
    dropout_rate = 0.1
    
    # Load data
    train_loader, valloader, test_loader = load_mnist_data(batch_size)
    
    # Create model
    model = LeHDCCNN(hdc_dimensions=hdc_dimensions, n_classes=10, dropout_rate=dropout_rate).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training and testing loop
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, epoch+1)
        val_loss, val_acc = test(model, valloader, criterion, device)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
        print(f'Test Loss: {val_loss:.4f}, Test Accuracy: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'cnn_lehdc_best.pth')
            print(f'Model saved with test accuracy: {val_acc:.2f}%')
    
    print(f'Best test accuracy: {best_val_acc:.2f}%')

if __name__ == '__main__':
    main()
