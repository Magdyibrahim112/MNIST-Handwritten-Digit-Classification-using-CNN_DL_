### MNIST-Handwritten-Digit-Classification-using-CNN_DL_

## 📌 Project Overview
This project implements Convolutional Neural Networks (CNNs) to classify handwritten digits using the MNIST dataset.  
Two different architectures are built and compared:

---

- Simple CNN
- Enhanced CNN (with Batch Normalization and Dropout)

## Dataset
The dataset used is the classic MNIST dataset of handwritten digits (0–9).

- 60,000 training images
- 10,000 testing images
- Image size: 28×28 grayscale


## Models

## 1️⃣ Simple CNN
- 2 Convolutional layers
- ReLU activation
- MaxPooling
- Fully connected layers

## 2️⃣ Enhanced CNN
- Convolutional layers
- Batch Normalization
- Dropout (0.25, 0.5)
- Fully connected layers

---

## Techniques Used
- Normalization
- Train/Validation Split (80/20)
- Cross Entropy Loss
- Adam Optimizer
- Batch Processing

---

### Factors tested:
- Architecture complexity
- Regularization (Dropout, BatchNorm)

---

## Results

| Model         | Accuracy |
|---------------|----------|
| Simple CNN    | 99.01%   |
| Enhanced CNN  | 99.20%   |

---

## Visualizations
The following plots are generated:
- Training vs Validation Accuracy
- Training vs Validation Loss

