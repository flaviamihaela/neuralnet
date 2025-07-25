# NeuralNet-Basics

## Overview

This repository contains foundational implementations of neural network models in Python, focusing on educational clarity and hands-on experimentation. The included notebooks demonstrate:
- A simple Perceptron for binary classification
- A single-layer neural network for multi-class tasks
- A multi-layer perceptron (MLP) for more complex classification (e.g., MNIST)

Each notebook is self-contained and designed for step-by-step exploration of neural network concepts, training, and evaluation.

## Key Features

- **Perceptron**: Classic binary classifier with adjustable learning rate and iterations.
- **Single Layer Neural Network**: Implements a single hidden layer, supports batch training, and visualizes loss/accuracy.
- **Multi-Layer Perceptron (MLP)**: Flexible architecture (number of layers, neurons per layer), supports categorical outputs, and tracks training metrics.
- **Visualization**: Built-in plotting for loss and accuracy curves.
- **MNIST Support**: Example code for loading and training on MNIST data.

## Dependencies

- Python 3.7+
- numpy
- matplotlib
- scikit-learn (for Perceptron data splitting)

Install dependencies with:
```bash
pip install numpy matplotlib scikit-learn
```

## Run and Build

All models are implemented as Jupyter notebooks. To run them:

1. Install Jupyter if you haven’t:
   ```bash
   pip install notebook
   ```
2. Start Jupyter in this directory:
   ```bash
   jupyter notebook
   ```
3. Open any of the following notebooks:
   - `Perceptron.ipynb`
   - `Single_Layer.ipynb`
   - `MLP.ipynb`
4. Run the cells sequentially. For the neural network notebooks, ensure you have the MNIST CSV files (or adapt the data loading code).

## Usage (Parameter Tuning)

Each model exposes parameters you can tune for experimentation:

- **Perceptron**
  - `lr`: Learning rate (default: 0.01)
  - `n_iters`: Number of training iterations (default: 1000)
  - Example:
    ```python
    clf = Perceptron(lr=0.01, n_iters=1000)
    ```

- **Single Layer Neural Network**
  - `batch`: Batch size (default: 64)
  - `lr`: Learning rate (default: 1e-3)
  - `epochs`: Number of epochs (default: 50)
  - Example:
    ```python
    NN = NeuralNetwork(X_train, y_train, batch=64, lr=1e-3, epochs=50)
    ```

- **MLP**
  - `L`: Number of hidden layers (default: 1)
  - `N_l`: Neurons per hidden layer (default: 128)
  - `batch_size`: Batch size for training (default: 8)
  - `epochs`: Number of epochs (default: 25)
  - `lr`: Learning rate (default: 1.0)
  - Example:
    ```python
    model = MLP(X_train, Y_train, L=2, N_l=64)
    model.train(batch_size=16, epochs=30, lr=0.5)
    ```

Adjust these parameters in the notebook cells to observe their effect on training speed, accuracy, and convergence. 