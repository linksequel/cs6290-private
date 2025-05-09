"""
Assignment 3: Adversarial Robustness of CNNs on MNIST
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import random
from adversarial_attacks_task2 import run_pgd_experiments
from transfer_attack_task3 import run_transferability_experiment
from adversarial_training_task4 import run_adversarial_training_experiment

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}") 

# Define CNN models
class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # First conv block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Second conv block
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class DeeperCNN(nn.Module):
    def __init__(self):
        super(DeeperCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class AlternativeCNN(nn.Module):
    def __init__(self):
        super(AlternativeCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x))))
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 32 * 7 * 7)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Data loading and preprocessing
def load_data(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        transform=transform, 
        download=True
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False, 
        transform=test_transform, 
        download=True
    )
    
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2
    )
    
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    return train_loader, test_loader

# Training function
def train(model, train_loader, optimizer, criterion, epochs=10):
    model.train()
    total_step = len(train_loader)
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        print(f'Epoch {epoch+1}, Loss: {running_loss/total_step:.4f}, Accuracy: {100 * correct / total:.2f}%')
    
    return model

# Evaluation function
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

# Save model function
def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Load model function
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()
    print(f"Model loaded from {path}")
    return model

# Main function for Task 1: Train and evaluate baseline model
def task1():
    print("Task 1: Training baseline CNN on MNIST")
    train_loader, test_loader = load_data()
    
    model = BaselineCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    model = train(model, train_loader, optimizer, criterion, epochs=10)
    
    # Evaluate the model
    accuracy = evaluate(model, test_loader)
    
    # Save the model
    save_model(model, 'models/baseline_cnn.pth')
    
    return model, accuracy

# Task 2: PGD attack
def task2(model=None, model_path=None):
    print("Task 2: Running PGD attack on MNIST")
    
    # Load data
    _, test_loader = load_data()
    
    # Load or use provided model
    if model is None:
        model = BaselineCNN().to(device)
        model = load_model(model, model_path or 'models/baseline_cnn.pth')
    
    # Run PGD experiments
    results = run_pgd_experiments(model, test_loader, device)
    
    return results

# Task 3: Transferability study
def task3():
    print("Task 3: Running transferability study on different CNN architectures")
    
    # Train deeper model if not already trained
    deeper_model = DeeperCNN().to(device)
    deeper_model_path = 'models/deeper_cnn.pth'
    try:
        deeper_model = load_model(deeper_model, deeper_model_path)
    except:
        print("Training deeper model...")
        train_loader, test_loader = load_data()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(deeper_model.parameters(), lr=0.001)
        deeper_model = train(deeper_model, train_loader, optimizer, criterion, epochs=10)
        evaluate(deeper_model, test_loader)
        save_model(deeper_model, deeper_model_path)
    
    # Train alternative model if not already trained
    alternative_model = AlternativeCNN().to(device)
    alternative_model_path = 'models/alternative_cnn.pth'
    try:
        alternative_model = load_model(alternative_model, alternative_model_path)
    except:
        print("Training alternative model...")
        train_loader, test_loader = load_data()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(alternative_model.parameters(), lr=0.001)
        alternative_model = train(alternative_model, train_loader, optimizer, criterion, epochs=10)
        evaluate(alternative_model, test_loader)
        save_model(alternative_model, alternative_model_path)
    
    # Run transferability experiment
    _, test_loader = load_data()
    model_paths = {
        'baseline': 'models/baseline_cnn.pth',
        'deeper': 'models/deeper_cnn.pth',
        'alternative': 'models/alternative_cnn.pth'
    }
    results = run_transferability_experiment(model_paths, test_loader, device)
    
    return results

# Task 4: Adversarial training
def task4():
    print("Task 4: Running adversarial training experiment")

    # Run adversarial training experiment
    standard_model, robust_model = run_adversarial_training_experiment(
        standard_model_path='models/baseline_cnn.pth',
        eps=0.1,
        adv_ratio=0.5,
        epochs=10
    )
    
    return standard_model, robust_model

if __name__ == "__main__":
    # Create directories to save models and results
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run assignments for CNN Adversarial Robustness')
    parser.add_argument('--task', type=int, default=0, help='Task to run (0 for all, 1-4 for specific task)')
    args = parser.parse_args()
    
    if args.task == 0 or args.task == 1:
        # Task 1: Train baseline CNN model
        print("\n========== Task 1: Training Baseline CNN ==========")
        baseline_model, baseline_accuracy = task1()
    
    if args.task == 0 or args.task == 2:
        # Task 2: PGD attack
        print("\n========== Task 2: PGD Attack ==========")
        if 'baseline_model' in locals():
            task2(model=baseline_model)
        else:
            task2(model_path='models/baseline_cnn.pth')
    
    if args.task == 0 or args.task == 3:
        # Task 3: Transferability study
        print("\n========== Task 3: Transferability Study ==========")
        task3()
    
    if args.task == 0 or args.task == 4:
        # Task 4: Adversarial training
        print("\n========== Task 4: Adversarial Training ==========")
        task4() 