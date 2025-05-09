# Adversarial Robustness of CNNs on MNIST

This repository contains the implementation for Assignment 3, which explores the adversarial robustness of CNNs on the MNIST dataset.

## Project Structure

- `main.py`: Main script that implements the CNN models and functions for training and evaluation
- `adversarial_attacks_task2.py`: Implementation of PGD attack and related functions for Task2
- `transfer_attack_task3.py`: Code for evaluating transferability of adversarial examples for Task3
- `adversarial_training_task4.py`: Implementation of adversarial training for Task4

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- matplotlib
- numpy
- tqdm

You can install the required packages using:

```bash
pip install torch torchvision matplotlib numpy tqdm
```

## How to Run

The code is organized into four main tasks that can be run individually or all at once:

### Running All Tasks

```bash
python main.py
```

### Running Individual Tasks

1. Train baseline CNN model:
```bash
python main.py --task 1
```

2. PGD attack implementation:
```bash
python main.py --task 2
```

3. Transferability study:
```bash
python main.py --task 3
```

4. Adversarial training:
```bash
python main.py --task 4
```

## Tasks Overview

### Task 1: MNIST Classifier
Implements a CNN classifier for MNIST with the following architecture:
- 2 convolutional layers with batch normalization and ReLU activation
- 2 fully connected layers with dropout
- Trained using Adam optimizer

### Task 2: PGD Attack
Implements the Projected Gradient Descent (PGD) attack with various epsilon values:
- Epsilon values: 0.01, 0.03, 0.05, 0.1
- Step size (alpha): 0.01
- Number of iterations: 40
- Random start from epsilon-ball

### Task 3: Transferability of Adversarial Examples
Evaluates the transferability of adversarial examples across different CNN architectures:
- Baseline CNN (Task 1)
- Deeper CNN (3 convolutional layers)
- Alternative CNN (different kernel sizes and activation function)

### Task 4: Adversarial Training
Implements adversarial training as a defense mechanism:
- 50% clean samples, 50% adversarial samples in each batch
- PGD attack with epsilon = 0.1 for generating adversarial examples
- Compares standard and robust models on both clean and adversarial examples

## Results

All results are saved in the `results/` directory, including:
- Model weights in `models/`
- Visualization of adversarial examples
- Accuracy plots under different attack scenarios
- Transferability study results
- Comparison of standard and robust models 