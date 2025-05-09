"""
Implementation of adversarial example transferability study for Assignment 3

Related to task3

"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from adversarial_attacks_task2 import pgd_attack
from common import BaselineCNN, DeeperCNN, AlternativeCNN, load_model, evaluate

def generate_adversarial_examples(source_model, test_loader, device, eps=0.1, alpha=0.01, iters=40):
    """
    Generate adversarial examples using the source model
    
    Args:
        source_model: Model to generate adversarial examples from
        test_loader: Test data loader
        device: Device to run on
        eps: Maximum perturbation
        alpha: Step size
        iters: Number of iterations
        
    Returns:
        List of (original_images, adversarial_images, labels)
    """
    source_model.eval()
    adv_examples = []
    
    for images, labels in tqdm(test_loader, desc=f'Generating adversarial examples (eps={eps})'):
        images, labels = images.to(device), labels.to(device)
        
        # Generate adversarial examples
        adv_images = pgd_attack(source_model, images, labels, eps=eps, alpha=alpha, iters=iters)
        
        # Save examples
        adv_examples.append((images, adv_images, labels))
        
        # Only generate a limited number of examples for efficiency
        if len(adv_examples) >= 10:
            break
    
    return adv_examples

def evaluate_transferability(source_model, target_models, test_loader, device, eps=0.1, alpha=0.01, iters=40):
    """
    Evaluate the transferability of adversarial examples
    
    Args:
        source_model: Model to generate adversarial examples from
        target_models: List of target models to test transferability
        test_loader: Test data loader
        device: Device to run on
        eps: Maximum perturbation
        alpha: Step size
        iters: Number of iterations
        
    Returns:
        Dictionary of results
    """
    # Generate adversarial examples using the source model
    adv_examples = generate_adversarial_examples(source_model, test_loader, device, eps, alpha, iters)
    
    # Evaluate clean accuracy for each model
    results = {}
    model_names = ['Source Model'] + [f'Target Model {i+1}' for i in range(len(target_models))]
    models = [source_model] + target_models
    
    for i, model in enumerate(models):
        model.eval()
        name = model_names[i]
        
        # Accuracy on clean examples
        clean_correct = 0
        # Accuracy on adversarial examples
        adv_correct = 0
        total = 0
        
        for original_images, adv_images, labels in adv_examples:
            batch_size = labels.size(0)
            total += batch_size
            
            # Evaluate on clean examples
            with torch.no_grad():
                outputs = model(original_images)
                _, predicted = torch.max(outputs.data, 1)
                clean_correct += (predicted == labels).sum().item()
            
            # Evaluate on adversarial examples
            with torch.no_grad():
                outputs = model(adv_images)
                _, predicted = torch.max(outputs.data, 1)
                adv_correct += (predicted == labels).sum().item()
        
        clean_accuracy = 100 * clean_correct / total
        adv_accuracy = 100 * adv_correct / total
        attack_success_rate = 100 - adv_accuracy
        
        print(f'{name}:')
        print(f'  Clean accuracy: {clean_accuracy:.2f}%')
        print(f'  Adversarial accuracy: {adv_accuracy:.2f}%')
        print(f'  Attack success rate: {attack_success_rate:.2f}%')
        
        results[name] = {
            'clean_accuracy': clean_accuracy,
            'adv_accuracy': adv_accuracy,
            'attack_success_rate': attack_success_rate
        }
    
    # Visualize results
    visualize_transferability_results(results)
    
    return results

def visualize_transferability_results(results):
    """
    Visualize transferability results
    
    Args:
        results: Dictionary of results from evaluate_transferability
    """
    model_names = list(results.keys())
    clean_accuracies = [results[name]['clean_accuracy'] for name in model_names]
    adv_accuracies = [results[name]['adv_accuracy'] for name in model_names]
    attack_success_rates = [results[name]['attack_success_rate'] for name in model_names]
    
    # Bar plot of accuracies
    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    index = np.arange(len(model_names))
    
    plt.bar(index, clean_accuracies, bar_width, label='Clean Accuracy')
    plt.bar(index + bar_width, adv_accuracies, bar_width, label='Adversarial Accuracy')
    
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    plt.title('Clean vs Adversarial Accuracy')
    plt.xticks(index + bar_width / 2, model_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/transferability_accuracy.png')
    plt.close()
    
    # Bar plot of attack success rates
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, attack_success_rates, color='crimson')
    plt.xlabel('Model')
    plt.ylabel('Attack Success Rate (%)')
    plt.title('Transferability of Adversarial Examples')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('results/transferability_success_rate.png')
    plt.close()
    
    # Create a table of results
    data = []
    for name in model_names:
        data.append([
            name,
            f"{results[name]['clean_accuracy']:.2f}%",
            f"{results[name]['adv_accuracy']:.2f}%",
            f"{results[name]['attack_success_rate']:.2f}%"
        ])
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    ax.axis('tight')
    
    table = ax.table(
        cellText=data,
        colLabels=['Model', 'Clean Accuracy', 'Adversarial Accuracy', 'Attack Success Rate'],
        loc='center',
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    plt.tight_layout()
    plt.savefig('results/transferability_table.png')
    plt.close()

def run_transferability_experiment(model_paths, test_loader, device, eps=0.1, alpha=0.01, iters=40):
    """
    Run transferability experiment using the specified models
    
    Args:
        model_paths: Dictionary of model paths {model_name: path}
        test_loader: Test data loader
        device: Device to run on
        eps: Maximum perturbation
        alpha: Step size
        iters: Number of iterations
    """
    # Load models
    baseline_model = BaselineCNN().to(device)
    baseline_model = load_model(baseline_model, model_paths['baseline'])
    
    deeper_model = DeeperCNN().to(device)
    deeper_model = load_model(deeper_model, model_paths['deeper'])
    
    alternative_model = AlternativeCNN().to(device)
    alternative_model = load_model(alternative_model, model_paths['alternative'])
    
    # Source model is the baseline model
    source_model = baseline_model
    # Target models are the deeper and alternative models
    target_models = [deeper_model, alternative_model]
    
    # Evaluate transferability
    results = evaluate_transferability(source_model, target_models, test_loader, device, eps, alpha, iters)
    
    return results 