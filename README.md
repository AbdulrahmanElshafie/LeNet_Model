# Convolutional Neural Network (CNN) LeNet Network Classifiers for CIFAR-10 and MNIST Datasets

This repository contains two Jupyter Notebooks that demonstrate the implementation of Convolutional Neural Networks (CNNs) for image classification tasks using TensorFlow and Keras.

## 1. CNN for CIFAR-10 Dataset (Notebook 1)

This notebook explores training a CNN to classify images from the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes (e.g., airplanes, cars, dogs).

### Key Steps:

- **Imports:** Loads essential libraries like TensorFlow, NumPy, and Matplotlib.
- **Data Loading:** Fetches the CIFAR-10 dataset using `tensorflow.keras.datasets.cifar10.load_data()`.
- **Preprocessing:**
  - Normalizes image pixel values to the range [0, 1] for better training performance.
  - Implements the `one_hot_encode` function to convert integer class labels (e.g., 0 for airplane) into one-hot encoded vectors, enabling the model to learn class probabilities.
- **Data Exploration:**
  - Prints shapes and sizes of training and testing data.
  - Visualizes the first 15 training images with their corresponding class labels using Matplotlib.

## 2. CNN for MNIST Dataset (Notebook 2)

This notebook follows a similar structure as Notebook 1, but trains a CNN to classify handwritten digits from the MNIST dataset, containing 70,000 28x28 grayscale images in 10 classes (digits 0-9).

## Further Considerations

- **Model Training:** The notebooks currently focus on data loading and preprocessing. You can extend them to build and train CNN models with appropriate layers (e.g., convolutional, pooling, flattening, dense) and activation functions. Consider using techniques like dropout and regularization to prevent overfitting.
- **Model Evaluation:** Implement metrics like accuracy, precision, recall, and F1-score to evaluate the performance of your CNN models on the testing data.
- **Hyperparameter Tuning:** Experiment with different learning rates, optimizer choices (e.g., Adam, SGD), and number of filters/neurons to potentially improve classification accuracy.
- **Visualizing Results:** Create confusion matrices to visualize how often the model predicted each class correctly or incorrectly.
