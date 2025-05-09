"""
Implementation of Adversarial Attacks for Assignment 3
Related to Task2
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Projected Gradient Descent (PGD) Attack
def pgd_attack(model, images, labels, eps=0.3, alpha=0.01, iters=40, random_start=True):
    """
    Implement PGD attack on the given model and images
    
    Args:
        model: Target model
        images: Original images
        labels: True labels
        eps: Maximum perturbation (epsilon)
        alpha: Step size
        iters: Number of iterations
        random_start: Whether to start with random noise
        
    Returns:
        Adversarial examples
    """
    images = images.clone().detach()
    labels = labels.clone().detach()
    
    # Initialize adversarial examples
    adv_images = images.clone().detach()
    
    # Random start (optional)
    if random_start:
        # Add uniform random noise in [-eps, eps]
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        # Clip to ensure we're still in [0, 1]
        adv_images = torch.clamp(adv_images, 0, 1)
    
    # Perform PGD attack
    for i in range(iters):
        adv_images.requires_grad = True
        
        # Forward pass
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Update adversarial images with gradient sign
        adv_images = adv_images.detach() + alpha * torch.sign(adv_images.grad.detach())
        
        # Project back to epsilon ball
        delta = torch.clamp(adv_images - images, -eps, eps)
        adv_images = torch.clamp(images + delta, 0, 1)
    
    return adv_images

# Evaluate model under PGD attack
def evaluate_under_attack(model, test_loader, device, eps=0.3, alpha=0.01, iters=40):
    """
    Evaluate model performance under PGD attack
    
    Args:
        model: Target model
        test_loader: Test data loader
        device: Device to run on
        eps: Maximum perturbation
        alpha: Step size
        iters: Number of iterations
        
    Returns:
        Accuracy on adversarial examples
    """
    model.eval()
    correct = 0
    total = 0
    
    for images, labels in tqdm(test_loader, desc=f'Evaluating under PGD attack (eps={eps})'):
        images, labels = images.to(device), labels.to(device)
        
        # Generate adversarial examples
        adv_images = pgd_attack(model, images, labels, eps=eps, alpha=alpha, iters=iters)
        
        # Evaluate on adversarial examples
        with torch.no_grad():
            outputs = model(adv_images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy under PGD attack (eps={eps}): {accuracy:.2f}%')
    return accuracy

# Visualize original and adversarial examples
def visualize_adversarial_examples(model, test_loader, device, eps=0.1, alpha=0.01, iters=40, num_examples=5):
    """
    Visualize original and adversarial examples with their predictions
    """
    model.eval()
    
    # Get a batch of data
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    
    # Generate adversarial examples
    adv_images = pgd_attack(model, images, labels, eps=eps, alpha=alpha, iters=iters)
    
    # Get predictions for both original and adversarial images
    with torch.no_grad():
        outputs = model(images)
        adv_outputs = model(adv_images)
        
        _, predicted = torch.max(outputs.data, 1)
        _, adv_predicted = torch.max(adv_outputs.data, 1)
    
    # Convert tensors to numpy for visualization
    images = images.cpu().numpy()
    adv_images = adv_images.cpu().numpy()
    labels = labels.cpu().numpy()
    predicted = predicted.cpu().numpy()
    adv_predicted = adv_predicted.cpu().numpy()
    
    # Plot examples
    plt.figure(figsize=(12, 8))
    for i in range(num_examples):
        # Original image
        plt.subplot(2, num_examples, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.title(f'Original\nTrue: {labels[i]}\nPred: {predicted[i]}')
        plt.axis('off')
        
        # Adversarial image
        plt.subplot(2, num_examples, num_examples + i + 1)
        plt.imshow(adv_images[i].reshape(28, 28), cmap='gray')
        plt.title(f'Adversarial\nTrue: {labels[i]}\nPred: {adv_predicted[i]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'results/task2/adversarial_examples_eps_{eps}.png')
    plt.close()
    
    # Print attack success rate
    success_rate = (predicted != adv_predicted).sum() / num_examples * 100
    print(f'Attack success rate: {success_rate:.2f}%')
    
    # Also visualize the perturbation
    plt.figure(figsize=(12, 4))
    for i in range(num_examples):
        plt.subplot(1, num_examples, i + 1)
        # Enhance the perturbation for better visibility
        perturbation = (adv_images[i] - images[i]).reshape(28, 28)
        plt.imshow(np.abs(perturbation) * 10, cmap='viridis')
        plt.title(f'Perturbation\nMagnitude: {np.max(np.abs(perturbation)):.4f}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'results/task2/perturbations_eps_{eps}.png')
    plt.close()

# Run PGD attack experiments with different epsilon values
def run_pgd_experiments(model, test_loader, device, eps_values=[0.01, 0.03, 0.05, 0.1], alpha=0.01, iters=40):
    """
    Run PGD attack experiments with different epsilon values
    """
    results = {}
    model.eval()
    
    # Evaluate on clean data first
    clean_accuracy = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating on clean data'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            clean_accuracy += (predicted == labels).sum().item()
    
    clean_accuracy = 100 * clean_accuracy / total
    print(f'Clean accuracy: {clean_accuracy:.2f}%')
    results['clean'] = clean_accuracy
    
    # Evaluate under different PGD attacks
    for eps in eps_values:
        accuracy = evaluate_under_attack(model, test_loader, device, eps=eps, alpha=alpha, iters=iters)
        results[eps] = accuracy
        
        # Visualize some examples
        visualize_adversarial_examples(model, test_loader, device, eps=eps, alpha=alpha, iters=iters)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(['clean'] + [str(eps) for eps in eps_values], [results['clean']] + [results[eps] for eps in eps_values], marker='o')
    plt.title('Model Accuracy under PGD Attack')
    plt.xlabel('Perturbation Budget (ε)')
    plt.ylabel('Accuracy (%)')
    plt.grid()
    plt.savefig('results/task2/pgd_attack_results.png')
    plt.close()
    
    return results 