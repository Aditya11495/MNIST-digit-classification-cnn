Handwritten Digit Recognition using CNN
**Project Overview**

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits (0–9) using the MNIST dataset.
The model is trained using TensorFlow/Keras and achieves high accuracy on unseen test data.

**The project demonstrates:**

Deep Learning fundamentals

Convolutional Neural Networks (CNN)

Image preprocessing techniques

Model evaluation and visualization

Confusion Matrix analysis

Real-time prediction on test images

**Dataset: MNIST**

The MNIST dataset consists of:

70,000 grayscale images

Image size: 28×28 pixels

10 output classes (digits 0–9)

60,000 training images

10,000 testing images

Each pixel value ranges from 0–255 and is normalized before training.

**Model Architecture**

The CNN architecture used in this project:

Conv2D (32 filters, 3×3, ReLU)

MaxPooling2D

Conv2D (64 filters, 3×3, ReLU)

MaxPooling2D

Flatten Layer

Dense (128 neurons, ReLU)

Dropout (0.5)

Output Layer (10 neurons, Softmax)

**Training Details**

Optimizer: Adam

Loss Function: sparse_categorical_crossentropy

Batch Size: 64

Epochs: 10

Validation Split: 20%

**Model Performance**

Training Accuracy: ~99%

Test Accuracy: ~98–99%

Low overfitting due to Dropout regularization

**Visualizations Included**

✔ Training vs Validation Accuracy
✔ Training vs Validation Loss
✔ Confusion Matrix
✔ Sample Predictions (True vs Predicted Labels)

**Sample Prediction Output**

The model randomly selects test images and predicts their digit class.
Example output:

True Label: 7
Predicted: 7

This demonstrates real handwritten digit recognition.

**Technologies Used**

Python

TensorFlow / Keras

NumPy

Matplotlib

Scikit-learn
