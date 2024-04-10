# ----------------------------------------------------------------------------
# ----------------- Importing Libraries  -------------------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.dataloader import default_collate
import torchvision
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Normalize, Resize
from torchvision.utils import make_grid
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models import vit_b_32
import random
import torchvision.utils as vutils
import os
import torch.optim as optim
from torch.nn.utils import prune
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.utils.prune as prune
import random



# Define the classes for CIFAR-10
classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


# ----------------- FIXED VALUES ------------------------------------
RANDOM_SEED = 42

# Define the transformation for CIFAR-10 dataset
transform = ToTensor()

# CIFAR-10 training dataset and DataLoader
trainset = CIFAR10(root="./data", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

#  saving original images too
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torchvision.utils as vutils
import random
import random
import numpy as np
import torch

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

# Define the transformation for CIFAR-10 dataset
transform = ToTensor()

# CIFAR-10 training dataset and DataLoader
trainset = CIFAR10(root="./data", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)


class MixUp(torch.nn.Module):
    """
    Mixup augmentation for training data.

    Parameters:
    sampling_method (int): Sampling method for mixup. 1: beta distribution, 2: uniform distribution
    num_classes (int): Number of classes in the dataset
    alpha (float): Alpha parameter for beta distribution
    uniform_range (list): Range for uniform distribution

    Returns:
    augmented_images (torch.Tensor): Input data after mixup
    augmented_labels (torch.Tensor): Target data after mixup
    """
    def __init__(
        self,
        sampling_method: int = 1,
        num_classes: int = 10,
        alpha: float = 1.0,
        uniform_range: list = [0.0, 1.0],
    ):
        super().__init__()
        self.sampling_method = sampling_method
        self.num_classes = num_classes
        self.alpha = alpha
        self.uniform_range = uniform_range

    # def __init__(
    #     self,
    #     sampling_method: int = 1,
    #     num_classes: int = 10,
    #     alpha: float = 1.0,
    #     uniform_range: list = [0.0, 1.0],
    # ):
    #     super().__init__()
    #     self.sampling_method = sampling_method
    #     self.num_classes = num_classes
    #     self.alpha = alpha
    #     self.uniform_range = uniform_range
    
    def __call__(self, inputs, labels):
        # Randomly select unique images from the dataset
        num_ims = len(inputs)
        all_indices = list(range(num_ims))
        random.shuffle(all_indices)
        selected_indices = all_indices[:num_ims]

        # Perform mixup for the selected images
        augmented_images = []
        augmented_labels = []
        original_images = []

        for idx in selected_indices:
            img = inputs[idx]
            label = labels[idx]
            img = img.unsqueeze(0)  # Add batch dimension
            label = torch.tensor([label])
            original_images.append(img)  # Store the original image

            # Sample lambda for mixup
            if self.sampling_method == 1:
                mixup_lambda = np.random.beta(self.alpha, self.alpha)
            elif self.sampling_method == 2:
                mixup_lambda = np.random.uniform(
                    self.uniform_range[0], self.uniform_range[1]
                )

            # Get a unique index for the second image
            second_idx = random.choice(all_indices)
            while second_idx == idx:
                second_idx = random.choice(all_indices)

            second_img = inputs[second_idx]
            second_label = labels[second_idx]
            second_img = second_img.unsqueeze(0)
            second_label = torch.tensor([second_label])

            augmented_img = mixup_lambda * img + (1 - mixup_lambda) * second_img
            augmented_label = (
                mixup_lambda * F.one_hot(label, num_classes=self.num_classes).float()
                + (1 - mixup_lambda)
                * F.one_hot(second_label, num_classes=self.num_classes).float()
            )

            augmented_images.append(augmented_img)
            augmented_labels.append(augmented_label)

        augmented_images = torch.cat(augmented_images, dim=0)
        augmented_labels = torch.cat(augmented_labels, dim=0)
        original_images = torch.cat(original_images, dim=0)

        return original_images, augmented_images, augmented_labels
    def __call__(self, num_ims=16):
        # Randomly select unique images from the dataset
        all_indices = list(range(len(trainset)))
        random.shuffle(all_indices)
        selected_indices = all_indices[:num_ims]

        # Perform mixup for the selected images
        augmented_images = []
        augmented_labels = []
        original_images = []

        for idx in selected_indices:
            img, label = trainset[idx]
            img = img.unsqueeze(0)  # Add batch dimension
            label = torch.tensor([label])
            original_images.append(img)  # Store the original image

            # Sample lambda for mixup
            if self.sampling_method == 1:
                mixup_lambda = np.random.beta(self.alpha, self.alpha)
            elif self.sampling_method == 2:
                mixup_lambda = np.random.uniform(
                    self.uniform_range[0], self.uniform_range[1]
                )

            # Get a unique index for the second image
            second_idx = random.choice(all_indices)
            while second_idx in selected_indices:
                second_idx = random.choice(all_indices)

            second_img, second_label = trainset[second_idx]
            second_img = second_img.unsqueeze(0)
            second_label = torch.tensor([second_label])

            augmented_img = mixup_lambda * img + (1 - mixup_lambda) * second_img
            augmented_label = (
                mixup_lambda * F.one_hot(label, num_classes=self.num_classes).float()
                + (1 - mixup_lambda)
                * F.one_hot(second_label, num_classes=self.num_classes).float()
            )

            augmented_images.append(augmented_img)
            augmented_labels.append(augmented_label)

        augmented_images = torch.cat(augmented_images, dim=0)
        augmented_labels = torch.cat(augmented_labels, dim=0)
        original_images = torch.cat(original_images, dim=0)

        return original_images, augmented_images, augmented_labels

    def save_output(
        self, original_images, augmented_images, augmented_labels, save_path="mixup.png"
    ):
        """
        Save the original, augmented images and labels as image files.

        Parameters:
        original_images (torch.Tensor): Original input images
        augmented_images (torch.Tensor): Augmented input images
        augmented_labels (torch.Tensor): Augmented input labels
        save_path (str): Path to save the visualization image
        """
        # Normalize the images to [-1, 1] range
        original_images = (original_images * 2.0) - 1.0
        augmented_images = (augmented_images * 2.0) - 1.0

        # Save the original images
        original_grid = vutils.make_grid(
            original_images, nrow=4, padding=2, normalize=True, value_range=(-1, 1)
        )
        vutils.save_image(original_grid, "original_mixup.png")

        # Save the augmented images
        augmented_grid = vutils.make_grid(
            augmented_images, nrow=4, padding=2, normalize=True, value_range=(-1, 1)
        )
        vutils.save_image(augmented_grid, save_path)

        # Save the labels to a text file
        label_file = save_path.rsplit(".", 1)[0] + "_labels.txt"
        with open(label_file, "w") as f:
            for label in augmented_labels:
                f.write(str(torch.argmax(label).item()) + "\n")


# Create an instance of MixUp
mixup = MixUp(sampling_method=1, num_classes=10, alpha=1.0, uniform_range=[0.0, 1.0])

# Apply mixup augmentation
original_images, augmented_images, augmented_labels = mixup(num_ims=16)

# Save the output
mixup.save_output(
    original_images, augmented_images, augmented_labels, save_path="mixup.png"
)


import torch.nn as nn
from torchvision.models import vit_b_32


class ViT(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.vit = vit_b_32(pretrained=True)

        # Freeze all layers in the pretrained model
        for param in self.vit.parameters():
            param.requires_grad = False

        # Replace the head with a new linear layer
        self.vit.heads.head = nn.Linear(
            self.vit.heads.head.in_features, num_classes
        )  # more efficient in terms of model size because it only replaces the final linear layer

    def forward(self, x):
        x = self.vit(x)
        return x

PRUNING_AMOUNT = 0.1


def apply_pruning(module, amount=PRUNING_AMOUNT):
    """Apply unstructured pruning based on the L1 norm of weights."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            prune.l1_unstructured(m, name="weight", amount=amount)


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_with_mixup(sampling_method, num_epochs=20):

    # Defining the data transformation for CIFAR-10
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            ),  # Normalize the images
        ]
    )

    # Load the CIFAR-10 dataset - train and test
    trainset = torchvision.datasets.CIFAR10(
        root="data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

    testset = torchvision.datasets.CIFAR10(
        root="data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model, loss function, and optimizer
    # Ensure the SimplifiedViT class is correctly initialized as per your modifications
    net = ViT().to(device)
    net.vit.heads.head.apply(initialize_weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(), lr=0.01, momentum=0.9
    )  # v2 - lr=0.001 brought very low results with SimplifiedViT v1 -> lr=0.01
    mixup = MixUp(alpha=1.0, sampling_method=sampling_method, seed=42)

    # v2 - Introduce a learning rate scheduler
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.1
    )  # Adjust learning rate every 5 epochs

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
            _, predicted = torch.max(outputs.data, 1)  # Get the predicted labels
            total += labels.size(0)
            correct += (
                (
                    lam * (predicted == targets_a).float()
                    + (1 - lam) * (predicted == targets_b).float()
                )
                .sum()
                .item()
            )

        # v4 - Prunning
        # Apply pruning at specified epochs and gradually increase the amount
        if epoch % 5 == 4:  # Example: Apply pruning every 5 epochs
            prune_amount = 0.05 + 0.05 * (
                epoch // 5
            )  # Increase pruning amount gradually
            apply_pruning(net, amount=prune_amount)
            print(f"Applied pruning with amount {prune_amount:.2f}")

        # v2 - Step the learning rate scheduler
        scheduler.step()

        train_acc.append(100 * correct / total)
        print(f"Epoch {epoch+1} - Training accuracy: {train_acc[-1]:.2f}%")

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
        print(f"Epoch {epoch+1} - Test accuracy: {test_acc[-1]:.2f}%")

    # Save the trained model
    model_path = os.path.join(".", f"model_sampling_{sampling_method}.pth")
    torch.save(net.state_dict(), model_path)
    print(f"Model with sampling method {sampling_method} saved to {model_path}")

    return train_acc, test_acc

if __name__ == "__main__":
    print("Training with sampling method 1 (beta distribution)")
    train_acc_1, test_acc_1 = train_with_mixup(sampling_method=1)

    print("Training with sampling method 2 (uniform distribution)")
    train_acc_2, test_acc_2 = train_with_mixup(sampling_method=2)

    # Report test set performance
    print("Test set performance for sampling method 1:")
    for epoch, acc in enumerate(test_acc_1):
        print(f"Epoch {epoch+1} - Test accuracy: {acc:.2f}%")

    print("Test set performance for sampling method 2:")
    for epoch, acc in enumerate(test_acc_2):
        print(f"Epoch {epoch+1} - Test accuracy: {acc:.2f}%")