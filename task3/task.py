import torch
import torchvision
import torch.nn.functional as activation_functions
from torch.utils.data import default_collate
import torch.nn as neural_network
import torch.optim as optimization
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.models import vit_b_32, ViT_B_32_Weights
from torch import nn, optim
from PIL import Image, ImageDraw
import numpy as np
import argparse
import os
from torch.nn import CrossEntropyLoss

# Function for training the model
import torch.utils.data as data


import torch
import torch.utils.data as data

# ------------------------- STEP 1: Set up environment
# Set random seed for reproducibility
np.random.seed(32)
torch.manual_seed(42)
GEN_SEED = torch.Generator().manual_seed(42)

# Augmentation class
import torch
import numpy as np
import torch.nn.functional as F


class MixUp:
    """
    MixUp Data Augmentation Class

    MixUp performs data augmentation by creating convex combinations of pairs of images and their labels,
    improving model generalization by encouraging linear behavior in-between training examples.

    Attributes:
    - mix_sampling_method (int): Determines the method for sampling the MixUp parameter λ.
        - 1: Sample λ from a Beta distribution with parameters (alpha, alpha).
        - 2: Sample λ uniformly from the range specified in 'uniform_range'.
    - alpha (float): The alpha parameter for the Beta distribution, relevant when mix_sampling_method is 1.
    - uniform_range (tuple of float): The range from which λ is uniformly sampled, relevant when mix_sampling_method is 2.
    - num_classes (int): The number of classes in the dataset, used for one-hot encoding the labels.

    Methods:
    - __call__(images, labels): Applies MixUp augmentation to a batch of images and labels.
    """

    def __init__(
        self, mix_sampling_method=1, alpha=0.2, uniform_range=(0.0, 1.0), num_classes=10
    ):
        self.mix_sampling_method = mix_sampling_method
        self.alpha = alpha
        self.uniform_range = uniform_range
        self.num_classes = num_classes

    def __call__(self, images, labels):
        """
        Apply MixUp augmentation to a batch of images and labels.

        Parameters:
        - images (Tensor): A batch of images.
        - labels (Tensor): Corresponding labels for the batch of images.

        Returns:
        - mixed_images (Tensor): Augmented images after applying MixUp.
        - mixed_labels (Tensor): Augmented labels after applying MixUp.
        """
        batch_size = images.size(0)
        # Generate MixUp lambda parameter based on the specified sampling method
        if self.mix_sampling_method == 1:
            lam = np.random.beta(self.alpha, self.alpha, size=batch_size)
        else:  # uniform sampling
            lam = np.random.uniform(
                self.uniform_range[0], self.uniform_range[1], size=batch_size
            )

        lam = torch.from_numpy(lam).float().to(images.device)
        lam = lam.view(batch_size, 1, 1, 1)
        index = torch.randperm(batch_size).to(images.device)

        mixed_images = lam * images + (1 - lam) * images[index, :]
        mixed_labels = self._mix_labels(labels, index, lam[:, 0, 0, 0])

        return mixed_images, mixed_labels

    def _mix_labels(self, labels, index, lam):
        """
        Mix labels using the same lambda parameter used for mixing images.

        Parameters:
        - labels (Tensor): A batch of labels.
        - index (Tensor): A tensor of shuffled indices.
        - lam (Tensor): The lambda parameter used for mixing.

        Returns:
        - mixed_labels (Tensor): A tensor of mixed labels.
        """
        one_hot_labels = F.one_hot(labels, num_classes=self.num_classes).float()
        return (
            lam.view(-1, 1) * one_hot_labels
            + (1 - lam.view(-1, 1)) * one_hot_labels[index]
        )


# Function for saving example images
def save_sample_images(
    image_transform,
    dataset_loader,
    category_names,
    save_directory,
    prediction_model=None,
    sample_count=16,
):
    # ------------------------- STEP 2: Prepare data for visualization
    data_iterator = iter(dataset_loader)
    x_batch, y_batch = next(data_iterator)
    while len(x_batch) < sample_count:
        extra_x, extra_y = next(data_iterator)
        x_batch = torch.cat((x_batch, extra_x), dim=0)
        y_batch = torch.cat((y_batch, extra_y), dim=0)

    # --------------------------------------- Substep 2.1: Perform prediction if a model is provided
    if prediction_model is not None:
        model_outputs = prediction_model(x_batch)
        _, predictions = torch.max(model_outputs, 1)
        true_labels = [category_names[label] for label in y_batch]
        predicted_labels = [category_names[label] for label in predictions]

    mean_values = torch.tensor(image_transform.mean).view(3, 1, 1)
    std_values = torch.tensor(image_transform.std).view(3, 1, 1)

    image_width = x_batch[0].shape[1]
    image_height = x_batch[0].shape[2]
    grid_columns = 6
    grid_rows = sample_count // grid_columns + (1 if sample_count % grid_columns else 0)
    image_grid = Image.new(
        "RGB", (image_width * grid_columns, image_height * grid_rows), color="white"
    )

    # ------------------------- STEP 3: Visualize and save images
    for i in range(sample_count):
        image_data = (x_batch[i] * std_values + mean_values).numpy()
        image_data = np.transpose(image_data, (1, 2, 0))
        img = Image.fromarray((image_data * 255).astype(np.uint8))

        # --------------------------------------- Substep 3.1: Annotate images with labels or predictions
        if prediction_model is None:
            label_list = [
                category_names[idx] + ":" + str(round(y_batch[i][idx].item(), 2))
                for idx in y_batch[i].nonzero()
            ]
            img_title = f"{', '.join(label_list)}"
        else:
            img_title = f"GT: {true_labels[i]}\nPred: {predicted_labels[i]}"
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), img_title, fill="white")

        # --------------------------------------- Substep 3.2: Compile images into a grid
        row_num, col_num = divmod(i, grid_columns)
        image_grid.paste(img, (col_num * image_width, row_num * image_height))

    # --------------------------------------- Substep 3.3: Save the grid image
    if not os.getcwd().split(os.sep)[-1].startswith("task"):
        save_directory = "task2/" + save_directory
    image_grid.save(save_directory)
    print(f"Image saved at {save_directory}")


import torch
import torch.nn as nn
import torch.utils.data as data

import torch
import torch.nn as nn
import torch.utils.data as data
import torchmetrics


# Function for splitting the data into train, validation, and test sets
# def split_data(data, labels, dev_ratio=0.8, val_ratio=0.1):

#     num_samples = len(data)
#     num_dev = int(num_samples * dev_ratio)
#     num_val = int(num_dev * val_ratio)

#     indices = torch.randperm(num_samples).tolist()
#     dev_indices = indices[:num_dev]
#     test_indices = indices[num_dev:]

#     dev_data = data[dev_indices]
#     dev_labels = labels[dev_indices]
#     test_data = data[test_indices]
#     test_labels = labels[test_indices]

#     train_indices = dev_indices[:-num_val]
#     val_indices = dev_indices[-num_val:]

#     train_data = dev_data[:-num_val]
#     train_labels = dev_labels[:-num_val]
#     val_data = dev_data[-num_val:]
#     val_labels = dev_labels[-num_val:]

#     return train_data, train_labels, val_data, val_labels, test_data, test_labels


def perform_training(
    train_loader,
    val_loader,
    vision_model,
    model_optimizer,
    loss_function,
    device,
    num_classes,
    num_epochs=20,
    sampling_method=1,
):

    auroc = torchmetrics.AUROC(
        task="multiclass", num_classes=num_classes, average="macro"
    )
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)

    vision_model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_index, (x_input, y_label) in enumerate(train_loader, 0):
            x_input, y_label = x_input.to(device), y_label.to(device)

            # MixUp augmentation
            mixup = MixUp(sampling_method)
            mixed_x, mixed_y = mixup(x_input, y_label)

            model_optimizer.zero_grad()

            model_output = vision_model(mixed_x)
            batch_loss = loss_function(model_output, mixed_y)
            batch_loss.backward()
            model_optimizer.step()

            total_loss += batch_loss.item()
            # Print after each batch (optional)
            print(
                f"Epoch: {epoch + 1}, Batch: {batch_index + 1}, Batch Loss: {batch_loss.item():.3f}"
            )

        print(
            f"Epoch: {epoch + 1}, Training Loss: {total_loss / len(train_loader):.3f}"
        )

        vision_model.eval()
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                val_output = vision_model(x_val)
                auroc.update(val_output, y_val)
                accuracy.update(val_output.argmax(dim=1), y_val)

        val_auc = auroc.compute()
        val_accuracy = accuracy.compute()
        print(
            f"Validation AUC-ROC: {val_auc:.3f}, Validation Accuracy: {val_accuracy:.3f}"
        )

        auroc.reset()
        accuracy.reset()

        vision_model.train()

    torch.save(vision_model.state_dict(), f"model_{sampling_method}.pth")
    print("Saved trained model")


def test_model(test_loader, vision_model, device):
    vision_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = vision_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy of the model on the test images: {accuracy}%")


if __name__ == "__main__":
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import torchvision.transforms as transforms

    # Update the transform to include resizing
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize the images to 224x224
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # from https://pytorch.org/vision/stable/models.html
        ]
    )
    # ...

    # Load CIFAR-10 train dataset and further split it into test and validation
    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    #  Split
    train_set, test_set = torch.utils.data.random_split(
        dataset, [0.8, 0.2], generator=GEN_SEED
    )
    train_set, val_set = torch.utils.data.random_split(
        train_set, [0.9, 0.1], generator=GEN_SEED
    )

    # DataLoaders
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(train_set, batch_size=32, shuffle=False)

    criterion = CrossEntropyLoss()

    num_epochs = 20

    for sampling_method in [1, 2]:
        # MixUp initialization
        mixup = MixUp(mix_sampling_method=sampling_method)

        # Model setup
        model = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
        model.heads = nn.Sequential(
            nn.Linear(model.heads[0].in_features, 10)
        )  # Adjusting for CIFAR-10
        model.to(device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        # Training
        labels = torch.tensor(
            [dataset.targets[i] for i in train_set.indices]
        )  # Define labels for the training set

        perform_training(
            train_loader,
            val_loader,
            model,
            optimizer,
            criterion,
            device,
            10,  # num_classes argument
            num_epochs,
            sampling_method,
        )

        # Testing
        test_model(test_loader, model, device)
        save_sample_images(
            transform,
            test_loader,
            category_names=[
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
            ],
            save_directory=f"result_{sampling_method}.png",
            prediction_model=model,
            sample_count=36,
        )
