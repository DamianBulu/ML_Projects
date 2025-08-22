MNIST Neural Network from Scratch

A complete implementation of a neural network for handwritten digit classification on the MNIST dataset, built entirely from scratch in Python without using machine learning frameworks.
Project Overview

This project demonstrates the fundamental concepts of neural networks through a clean implementation of:

    Feedforward propagation with ReLU activation

    Backpropagation algorithm

    Stochastic Gradient Descent (SGD) optimization

    Cross-entropy loss function

    Hyperparameter tuning with random search

Architecture

The neural network consists of:

    Input layer: 784 neurons (28x28 pixels)

    Hidden layer: 300 neurons with ReLU activation

    Output layer: 10 neurons with Softmax activation (digit classes 0-9)

Key Features

    Modular design with separate components for data processing, model definition, training, and visualization

    Random search for hyperparameter optimization

    Model persistence with parameter saving/loading

    Training monitoring with loss and accuracy tracking

    Comprehensive evaluation on test dataset

Results

The implemented neural network achieves 94.82% accuracy on the MNIST test set, demonstrating the effectiveness of the from-scratch implementation.

Project Structure

MNIST_Classifier_From_Scratch/
--->model.py                 # Neural network architecture
---> data_processor.py       # Data loading and preprocessing
--->trainer.py               # Training loop and evaluation
--->visualizer.py            # Results visualization
--->hyperparameter_tuner.py  # Random search implementation
--->main.py                  # Main execution script
--->requirements.txt         # Project dependencies

Installation

    Clone the repository

    Install required dependencies:pip install numpy matplotlib scikit-learn

Usage

Run the main script to train and evaluate the model:python main.py

The script will automatically:

    Download and preprocess the MNIST dataset

    Perform hyperparameter tuning using random search

    Train the final model with optimized parameters

    Evaluate performance on test data

    Generate visualizations of training progress

Technical Details

    Initialization: Xavier/Glorot initialization for stable training

    Optimization: SGD with learning rate scheduling

    Regularization: Implicit through stochastic gradient descent

    Evaluation: Standard MNIST test set (10,000 samples)

Modules

    model.py: Contains the NeuralNetwork class with forward and backward propagation

    data_processor.py: Handles data loading and preprocessing using sklearn

    trainer.py: Implements the training loop with evaluation metrics

    visualizer.py: Provides plotting functions for training history

    hyperparameter_tuner.py: Performs random search for parameter optimization