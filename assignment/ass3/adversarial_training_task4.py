"""
Implementation of Adversarial Training for Assignment 3
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from adversarial_attacks_task2 import pgd_attack, evaluate_under_attack

def adversarial_training(model, train_loader, test_loader, device, 
                         eps=0.1, alpha=0.01, iters=10, 
                         adv_ratio=0.5, epochs=10, 
                         learning_rate=0.001):
    """
    Implement adversarial training
    
    Args:
        model: Model to train
        train_loader: Training data loader
        test_loader: Test data loader
        device: Device to run on
        eps: Maximum perturbation for generating adversarial examples
        alpha: Step size for PGD
        iters: Number of iterations for PGD
        adv_ratio: Ratio of adversarial examples in each batch
        epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        
    Returns:
        Trained model
    """
    from main import evaluate

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Keep track of losses and accuracies
    train_losses = []
    train_accuracies = []
    clean_test_accuracies = []
    adv_test_accuracies = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')):
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)
            
            # Generate adversarial examples for a portion of the batch
            if adv_ratio > 0:
                # Number of adversarial examples to generate
                num_adv = int(batch_size * adv_ratio)
                
                # Split the batch
                adv_images = images[:num_adv]
                adv_labels = labels[:num_adv]
                clean_images = images[num_adv:]
                clean_labels = labels[num_adv:]
                
                # Generate adversarial examples
                adv_images = pgd_attack(model, adv_images, adv_labels, eps=eps, alpha=alpha, iters=iters)
                
                # Combine clean and adversarial examples
                combined_images = torch.cat([clean_images, adv_images], dim=0)
                combined_labels = torch.cat([clean_labels, adv_labels], dim=0)
            else:
                # Use only clean examples
                combined_images = images
                combined_labels = labels
            
            # Forward pass
            outputs = model(combined_images)
            loss = criterion(outputs, combined_labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += combined_labels.size(0)
            correct += (predicted == combined_labels).sum().item()
        
        # Calculate epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')
        
        # Evaluate on clean test data
        model.eval()
        clean_accuracy = evaluate(model, test_loader)
        clean_test_accuracies.append(clean_accuracy)
        
        # Evaluate on adversarial test data
        adv_accuracy = evaluate_under_attack(model, test_loader, device, eps=eps, alpha=alpha, iters=iters)
        adv_test_accuracies.append(adv_accuracy)
        
        model.train()
    
    # Plot training curves
    plot_training_curves(train_losses, train_accuracies, clean_test_accuracies, adv_test_accuracies)
    
    return model

def plot_training_curves(train_losses, train_accuracies, clean_test_accuracies, adv_test_accuracies):
    """
    Plot training curves
    
    Args:
        train_losses: List of training losses
        train_accuracies: List of training accuracies
        clean_test_accuracies: List of clean test accuracies
        adv_test_accuracies: List of adversarial test accuracies
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot training loss
    ax1.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-', color='b', label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(range(1, len(train_accuracies) + 1), train_accuracies, marker='o', linestyle='-', color='b', label='Training Accuracy')
    ax2.plot(range(1, len(clean_test_accuracies) + 1), clean_test_accuracies, marker='s', linestyle='-', color='g', label='Clean Test Accuracy')
    ax2.plot(range(1, len(adv_test_accuracies) + 1), adv_test_accuracies, marker='^', linestyle='-', color='r', label='Adversarial Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracies')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('results/adversarial_training_curves.png')
    plt.close()

def compare_models(standard_model, robust_model, test_loader, device, eps_values=[0.01, 0.03, 0.05, 0.1]):
    """
    Compare standard and robust models
    
    Args:
        standard_model: Standard model (trained normally)
        robust_model: Robust model (trained adversarially)
        test_loader: Test data loader
        device: Device to run on
        eps_values: Epsilon values to test
    """
    from main import evaluate

    # Evaluate on clean data
    print("Evaluating on clean data:")
    standard_clean_acc = evaluate(standard_model, test_loader)
    robust_clean_acc = evaluate(robust_model, test_loader)
    
    # Evaluate on adversarial data with different epsilon values
    standard_adv_accs = []
    robust_adv_accs = []
    
    for eps in eps_values:
        print(f"\nEvaluating with eps={eps}:")
        
        print("Standard model:")
        standard_adv_acc = evaluate_under_attack(standard_model, test_loader, device, eps=eps)
        standard_adv_accs.append(standard_adv_acc)
        
        print("Robust model:")
        robust_adv_acc = evaluate_under_attack(robust_model, test_loader, device, eps=eps)
        robust_adv_accs.append(robust_adv_acc)
    
    # Plot comparison
    plot_model_comparison(standard_clean_acc, robust_clean_acc, standard_adv_accs, robust_adv_accs, eps_values)

def plot_model_comparison(standard_clean_acc, robust_clean_acc, standard_adv_accs, robust_adv_accs, eps_values):
    """
    Plot comparison between standard and robust models
    
    Args:
        standard_clean_acc: Clean accuracy of standard model
        robust_clean_acc: Clean accuracy of robust model
        standard_adv_accs: List of adversarial accuracies for standard model
        robust_adv_accs: List of adversarial accuracies for robust model
        eps_values: Epsilon values tested
    """
    # Combine clean and adversarial accuracies
    standard_accs = [standard_clean_acc] + standard_adv_accs
    robust_accs = [robust_clean_acc] + robust_adv_accs
    
    # X-axis labels
    x_labels = ['Clean'] + [f'ε={eps}' for eps in eps_values]
    
    # Plot bar chart
    bar_width = 0.35
    index = np.arange(len(x_labels))
    
    plt.figure(figsize=(12, 6))
    plt.bar(index, standard_accs, bar_width, label='Standard Model', color='skyblue')
    plt.bar(index + bar_width, robust_accs, bar_width, label='Robust Model', color='salmon')
    
    plt.xlabel('Perturbation Budget')
    plt.ylabel('Accuracy (%)')
    plt.title('Standard vs. Robust Model Accuracy')
    plt.xticks(index + bar_width / 2, x_labels)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png')
    plt.close()
    
    # Compute accuracy improvement
    improvements = np.array(robust_adv_accs) - np.array(standard_adv_accs)
    
    # Plot accuracy improvement
    plt.figure(figsize=(10, 6))
    plt.bar(eps_values, improvements, color='green')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    plt.xlabel('Perturbation Budget (ε)')
    plt.ylabel('Accuracy Improvement (%)')
    plt.title('Robustness Improvement from Adversarial Training')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('results/robustness_improvement.png')
    plt.close()
    
    # Create a table of results
    data = [
        ['Clean', f"{standard_clean_acc:.2f}%", f"{robust_clean_acc:.2f}%", f"{robust_clean_acc-standard_clean_acc:.2f}%"]
    ]
    
    for i, eps in enumerate(eps_values):
        data.append([
            f'ε={eps}',
            f"{standard_adv_accs[i]:.2f}%",
            f"{robust_adv_accs[i]:.2f}%",
            f"{improvements[i]:.2f}%"
        ])
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    ax.axis('tight')
    
    table = ax.table(
        cellText=data,
        colLabels=['Test Data', 'Standard Model', 'Robust Model', 'Improvement'],
        loc='center',
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    plt.tight_layout()
    plt.savefig('results/robustness_table.png')
    plt.close()

def run_adversarial_training_experiment(standard_model_path=None, eps=0.1, adv_ratio=0.5, epochs=10):
    """
    Run adversarial training experiment
    
    Args:
        standard_model_path: Path to a standard model, if None a new model will be trained
        eps: Maximum perturbation for adversarial training
        adv_ratio: Ratio of adversarial examples in each batch
        epochs: Number of training epochs
    """
    from main import BaselineCNN, evaluate, load_data, save_model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    train_loader, test_loader = load_data()
    
    # Standard model (either load or train)
    standard_model = BaselineCNN().to(device)
    if standard_model_path:
        standard_model.load_state_dict(torch.load(standard_model_path))
        print(f"Loaded standard model from {standard_model_path}")
    else:
        print("Training standard model from scratch...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(standard_model.parameters(), lr=0.001)
        
        # Train the standard model
        for epoch in range(epochs):
            standard_model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, (images, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} (Standard Model)')):
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                outputs = standard_model(images)
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
            
            print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%')
        
        # Save the standard model
        save_model(standard_model, 'models/standard_baseline_cnn.pth')
    
    # Evaluate standard model on clean data
    print("Evaluating standard model on clean data:")
    standard_model.eval()
    standard_clean_acc = evaluate(standard_model, test_loader)
    
    # Evaluate standard model on adversarial data
    print(f"Evaluating standard model on adversarial data (eps={eps}):")
    standard_adv_acc = evaluate_under_attack(standard_model, test_loader, device, eps=eps)
    
    # Train robust model with adversarial training
    print("\nTraining robust model with adversarial training...")
    robust_model = BaselineCNN().to(device)
    robust_model = adversarial_training(
        robust_model, train_loader, test_loader, device,
        eps=eps, adv_ratio=adv_ratio, epochs=epochs
    )
    
    # Save the robust model
    save_model(robust_model, f'models/robust_baseline_cnn_eps_{eps}.pth')
    
    # Compare standard and robust models
    print("\nComparing standard and robust models:")
    compare_models(standard_model, robust_model, test_loader, device)
    
    return standard_model, robust_model 