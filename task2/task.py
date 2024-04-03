import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.utils.prune as prune
from torchvision.transforms import Resize
from models import SimplifiedViT
from data import MixUp



def apply_pruning(module, amount=0.1):
    """ Apply unstructured pruning based on the L1 norm of weights. """
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            prune.l1_unstructured(m, name='weight', amount=amount)


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_with_mixup(sampling_method, num_epochs=20):
    
    # Defining the data transformation for CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the images
    ])

    # Load the CIFAR-10 dataset - train and test
    trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model, loss function, and optimizer
    # Ensure the SimplifiedViT class is correctly initialized as per your modifications
    net = SimplifiedViT().to(device)
    net.apply(initialize_weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)  #v2 - lr=0.001 brought very low results with SimplifiedViT v1 -> lr=0.01
    mixup = MixUp(alpha=1.0, sampling_method=sampling_method, seed=42)
    
    # v2 - Introduce a learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Adjust learning rate every 5 epochs


    train_acc, test_acc = [], []  # Initialize accuracy lists

    for epoch in range(num_epochs):
        running_loss, correct, total = 0.0, 0, 0

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
            correct += (lam * (predicted == targets_a).float() + (1 - lam) * (predicted == targets_b).float()).sum().item()
            
        # v4 - Prunning
        # Apply pruning at specified epochs and gradually increase the amount
        if epoch % 5 == 4:  # Example: Apply pruning every 5 epochs
            prune_amount = 0.05 + 0.05 * (epoch // 5)  # Increase pruning amount gradually
            apply_pruning(net, amount=prune_amount)
            print(f'Applied pruning with amount {prune_amount:.2f}')
        
        # v2 - Step the learning rate scheduler
        scheduler.step()
        
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
