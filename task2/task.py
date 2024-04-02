import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from models import ViT
from data import MixUp

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_with_mixup(sampling_method, num_epochs=20):
    
    # Defining the data transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # necessary for the ViT model
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the CIFAR-10 dataset - train and test
    trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

    # Define the classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model, loss function, and optimizer
    net = ViT().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    mixup = MixUp(alpha=1.0, sampling_method=sampling_method, seed=42)

    train_acc = []
    test_acc = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        # Training loop
        net.train()  # Set the model to training mode
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, targets_a, targets_b, lam = mixup(inputs, labels)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1) # Get the predicted labels
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc.append(100 * correct / total)
        print(f'Epoch {epoch+1} - Training accuracy: {train_acc[-1]:.2f}%')

        # Test loop
        net.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_acc.append(100 * correct / total)
        print(f'Epoch {epoch+1} - Test accuracy: {test_acc[-1]:.2f}%')

    # Save the trained model
    model_path = os.path.join('.', f'model_sampling_{sampling_method}.pth')
    torch.save(net.state_dict(), model_path)
    print(f'Model with sampling method {sampling_method} saved to {model_path}')

    return train_acc, test_acc

if __name__ == "__main__":
    print('Training with sampling method 1 (beta distribution)')
    train_acc_1, test_acc_1 = train_with_mixup(sampling_method=1)

    print('Training with sampling method 2 (uniform distribution)')
    train_acc_2, test_acc_2 = train_with_mixup(sampling_method=2)

    # Report test set performance
    print('Test set performance for sampling method 1:')
    for epoch, acc in enumerate(test_acc_1):
        print(f'Epoch {epoch+1} - Test accuracy: {acc:.2f}%')

    print('Test set performance for sampling method 2:')
    for epoch, acc in enumerate(test_acc_2):
        print(f'Epoch {epoch+1} - Test accuracy: {acc:.2f}%')
