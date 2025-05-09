"""
Assignment 3: Adversarial Robustness of CNNs on MNIST
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import argparse
from common import (
    BaselineCNN, DeeperCNN, AlternativeCNN, 
    load_data, evaluate, save_model, load_model,
    set_seed, device
)
from adversarial_attacks_task2 import run_pgd_experiments
from transfer_attack_task3 import run_transferability_experiment
from adversarial_training_task4 import run_adversarial_training_experiment

# Training function
def train(model, train_loader, optimizer, criterion, epochs=10):
    model.train()
    total_step = len(train_loader)
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(tqdm(train_loader, desc=f' Epoch {epoch+1}/{epochs}')):
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

# Main function for Task 1: Train and evaluate a baseline model
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
    # Set random seeds
    set_seed()
    print(f"Using device: {device}")

    # Create directories to save models and results
    os.makedirs('models', exist_ok=True)
    os.makedirs('results/task2', exist_ok=True)
    os.makedirs('results/task3', exist_ok=True)
    os.makedirs('results/task4', exist_ok=True)
    
    # Parse command line arguments
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