# 24_ViT_class

**Project Title:** Multiple-class classification with Vision Transformers \\ Applied Deep Learning Coursework - COMP0197

**Description**

This repository contains the individual coursework for the COMP0197: Applied Deep Learning module. It consists of three tasks designed to assess your understanding and application of deep learning principles using PyTorch. The coursework emphasizes implementing deep learning algorithms across different domains, including polynomial function modeling, advanced convolutional network architectures for image classification, and comprehensive ablation studies.

# Environment Setup

To ensure compatibility, update your operating system and configure the following Conda environment for PyTorch:

```bash
conda create -n comp0197-cw1-pt -c pytorch python=3.12 pytorch=2.2 torchvision=0.17
```

# Repository Structure**

* `task1/`: Contains the `task.py` script for implementing polynomial function fitting using stochastic minibatch gradient descent.
* `task2/`: Includes the `task.py` script for the Vision Transformer (ViT) classification task on the CIFAR-10 dataset, exploring various model configurations to assess their impact on image classification performance.
* `task3/`: Houses the `task.py` script for conducting an ablation study, focusing on the effects of different λ sampling methods used in Task 2's MixUp data augmentation on model performance.

# Tasks descriptions

## Task 1: Stochastic Gradient Descent for Polynomial Regression

Task 1 focuses on polynomial function modeling using stochastic minibatch gradient descent for linear models. Participants implement a vectorized polynomial function, polynomial_fun, and two optimization functions: fit_polynomial_ls for least squares fitting, and fit_polynomial_sgd for optimization via stochastic minibatch gradient descent. The task involves generating datasets with Gaussian noise, fitting polynomial models using both methods, and evaluating their performance in terms of accuracy and efficiency.

## Task 2: Advanced ViT Configurations with MixUp Augmentation

Task 2 focuses on training a depth-wise separable convolution network, specifically employing various configurations of the VisionTransformer (ViT) architecture, for the task of classifying images from the CIFAR-10 dataset. A key feature of this task is the incorporation of the MixUp data augmentation technique, which is expected to enhance model robustness and generalization by blending images and their labels in a manner that encourages the model to learn more general representations. The task provides an opportunity to assess the impact of MixUp, alongside the exploration of VisionTransformer models under different architectural adjustments:

* **Version 2:** Introduces a learning rate scheduler, an embedding size of 128, 2 attention heads, an expansion factor of 4, a dropout rate of 0.2, and 3 transformer blocks, all while employing MixUp data augmentation to study its effect on model training dynamics and performance.
* **Version 3:** Expands the embedding size to 256, increases to 4 attention heads, maintains an expansion factor of 4 and a dropout rate of 0.2, and consists of 3 transformer blocks, integrating MixUp to examine its influence on the model's ability to generalize from augmented data representations.
* **Version 4:** Incorporates weight initialization techniques, model pruning, and image resizing strategies to boost performance, with MixUp augmentation playing a role in testing the resilience of these architectural enhancements against overfitting and enhancing model adaptability.
* **Version 5:** Elevates the embedding size to 768 and the number of attention heads to 8, pushing the model's capacity boundaries, while the inclusion of MixUp aims to scrutinize how such a scaled-up model benefits from augmented data mixing in terms of learning efficiency and predictive accuracy. 

## Task 3: Lambda Sampling Impact in MixUp Ablation Study

Task 3 focuses on conducting an ablation study to explore the effects of different λ sampling methods used in the MixUp data augmentation on the performance of models trained on the CIFAR-10 dataset. The task involves a detailed examination of how varying the approach to sampling λ — either from a beta distribution or uniformly from a predefined range — influences model accuracy, loss values, and other relevant performance metrics. Participants are required to split the dataset into development and holdout test sets, further subdividing the development set for training and validation, and then train models using both λ sampling methods. The objective is to provide a comprehensive comparison of the results, offering insights into the impact of these data augmentation techniques on model effectiveness, with an emphasis on understanding which method contributes more positively to model generalization and performance on unseen data.


# Key Skills and Methods**

* Polynomial Function Modeling: Develop vectorized polynomial functions and a least squares solver using PyTorch.
* Gradient Descent Optimization: Apply minibatch stochastic gradient descent for model fitting.
* VisionTransformer for Image Classification: Train VisionTransformer models with different configurations to classify CIFAR-10 images, exploring the impact of various architectural changes.
* Custom Data Augmentation: Create a MixUp data augmentation class specifically for image data, enhancing model generalization.
* Ablation Study: Evaluate model modifications through a systematic comparison of performance metrics.
* Conda Environment Management: Set up and manage Conda environments to handle project dependencies.

# Usage

To execute a task, navigate to the respective task directory and run the `task.py` script within the specified Conda environment:

```
python task.py
```

Ensure all required classes, functions, and supporting files as detailed in the coursework description are correctly implemented and present within each task directory.
