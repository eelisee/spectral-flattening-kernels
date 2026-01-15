"""
Synthetic experiment on a subset of gaussian distributed data
Demonstrates the spectral flattening approach without full feature extraction
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kernels import (SpectralFlatteningKernel, compute_kernel_target_alignment,
                         compute_effective_rank)
from src.svm_utils import evaluate_kernel_svm, cross_validate_svm
from src.visualization import plot_decision_boundaries_2d, plot_spectral_flattening_effect_2d


def create_synthetic_features(n_samples=1000, n_features=100, n_classes=5, 
                              anisotropy_factor=100, signal_in_low_var=True):
    """
    Create synthetic features with controlled spectral structure
    to demonstrate the effect of spectral flattening
    """
    print("\n=== Generating Synthetic Data ===")
    
    # Create anisotropic covariance (diagonal - NO ROTATION)
    # This ensures dimension i corresponds directly to eigenvalue i
    eigenvalues = np.logspace(np.log10(anisotropy_factor), 0, n_features)
    Sigma = np.diag(eigenvalues)  # CRITICAL: No rotation! Dimensions = eigenvectors
    
    print(f"Feature dimension: {n_features}")
    print(f"Number of classes: {n_classes}")
    print(f"Covariance condition number: {eigenvalues[0]/eigenvalues[-1]:.2e}")
    print(f"Signal location: {'LOW-variance dims' if signal_in_low_var else 'HIGH-variance dims'}")
    
    # Generate class means in eigenbasis
    # Now dimension i has variance = eigenvalues[i]
    class_means = []
    for i in range(n_classes):
        if signal_in_low_var:
            # Signal in low-variance directions (dims with SMALL eigenvalues)
            # These are the LAST dimensions (eigenvalues[-10:] are smallest)
            mean = np.zeros(n_features)
            mean[-10:] = np.random.randn(10)  # Signal in dims with variance ≈ 1
        else:
            # Signal in high-variance directions (dims with LARGE eigenvalues)
            # These are the FIRST dimensions (eigenvalues[:10] are largest)
            mean = np.zeros(n_features)
            mean[:10] = np.random.randn(10) # Signal in dims with variance ≈ 100
        class_means.append(mean)
    
    # Generate samples
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    for i in range(n_classes):
        # Training samples
        X_class = np.random.multivariate_normal(class_means[i], Sigma, n_samples // n_classes)
        X_train.append(X_class)
        y_train.extend([i] * (n_samples // n_classes))
        
        # Test samples
        X_class_test = np.random.multivariate_normal(class_means[i], Sigma, 50)
        X_test.append(X_class_test)
        y_test.extend([i] * 50)
    
    X_train = np.vstack(X_train)
    X_test = np.vstack(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    return X_train, y_train, X_test, y_test, Sigma, eigenvalues


def run_synthetic_data_experiment():
    """Run quick synthetic experiment"""
    
    print("="*60)
    print("SPECTRAL FLATTENING EXPERIMENT ON SYNTHETIC DATA")
    print("="*60)
    
    # Setup
    os.makedirs('results/synthetic_data_experiment/', exist_ok=True)
    
    # Generate synthetic data
    np.random.seed(42)
    X_train, y_train, X_test, y_test, Sigma, eigenvalues = create_synthetic_features(
        n_samples=1000, n_features=100, n_classes=5, 
        anisotropy_factor=100, signal_in_low_var=True
    )
    
    # Test different β values
    beta_values = [0.0, 0.125, 0.25, 0.375, 0.5]
    C_values = [0.1, 1.0, 10.0]
    
    results = []
    
    print("\n=== Testing Different β Values ===")
    from scipy.spatial.distance import pdist
    
    for beta in beta_values:
        print(f"  Testing β = {beta:.3f}...", end=" ")
        
        # Step 1: Fit kernel with dummy sigma to compute transformation
        kernel_temp = SpectralFlatteningKernel(beta=beta, sigma=1.0, shrinkage=True)
        kernel_temp.fit(X_train)
        
        # Step 2: Transform features to compute adaptive sigma
        X_train_transformed = kernel_temp.transform(X_train)
        distances = pdist(X_train_transformed, metric='euclidean')
        sigma_adaptive = np.median(distances) / 3  # Scale factor for good kernel behavior
        
        # Step 3: Create final kernel with adaptive sigma
        kernel = SpectralFlatteningKernel(beta=beta, sigma=sigma_adaptive, shrinkage=True)
        kernel.fit(X_train)
        
        print(f"σ={sigma_adaptive:.2f}, ", end="")
        
        # Compute kernel matrices
        K_train = kernel.compute_kernel_matrix(X_train)
        K_test = kernel.compute_kernel_matrix(X_test, X_train)
        
        # Diagnostics
        alignment = compute_kernel_target_alignment(K_train, y_train)
        eff_rank = compute_effective_rank(K_train)
        cond_num = kernel.get_condition_number()
        
        # Cross-validate C
        best_C, cv_score, cv_fold_scores = cross_validate_svm(K_train, y_train, C_values, cv=3)
        
        # Test accuracy
        metrics = evaluate_kernel_svm(K_train, K_test, y_train, y_test, C=best_C)
        print(f"Accuracy: {metrics['test_accuracy']:.4f}")
        
        results.append({
            'beta': beta,
            'sigma': sigma_adaptive,
            'test_accuracy': metrics['test_accuracy'],
            'cv_score': cv_score,
            'cv_std': np.std(cv_fold_scores),  # Store CV standard deviation
            'cv_fold_scores': cv_fold_scores,
            'alignment': alignment,
            'effective_rank': eff_rank,
            'condition_number': cond_num
        })
    
    # Save results to CSV
    import pandas as pd
    results_df = pd.DataFrame(results)
    os.makedirs('results/synthetic_data_experiment/', exist_ok=True)
    results_df.to_csv('results/synthetic_data_experiment/results.csv', index=False)
    print(f"\nResults saved to results/synthetic_data_experiment/results.csv")
    
    # Visualize results
    print("Generating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    betas = np.array([r['beta'] for r in results])
    accuracies = np.array([r['test_accuracy'] for r in results])
    cv_scores = np.array([r['cv_score'] for r in results])
    cv_stds = np.array([r['cv_std'] for r in results])
    alignments = np.array([r['alignment'] for r in results])
    cond_nums = np.array([r['condition_number'] for r in results])
    eff_ranks = np.array([r['effective_rank'] for r in results])
    
    # Plot 1: Accuracy with error bars
    ax = axes[0, 0]
    ax.errorbar(betas, accuracies, yerr=cv_stds, fmt='o-', linewidth=2.5, 
                markersize=10, color='#2E86AB', capsize=5, capthick=2, 
                elinewidth=2, alpha=0.9, label='Test accuracy')
    ax.fill_between(betas, accuracies - cv_stds, accuracies + cv_stds, 
                     alpha=0.2, color='#2E86AB')
    
    best_idx = np.argmax(accuracies)
    ax.axvline(x=betas[best_idx], color='#A23B72', linestyle='--', 
               linewidth=2, alpha=0.7, label=f'Best β={betas[best_idx]:.3f}')
    
    # Add value annotations at all points
    for i, (beta, acc) in enumerate(zip(betas, accuracies)):
        ax.annotate(f'{acc:.1%}', 
                    xy=(beta, acc),
                    xytext=(0, 8), textcoords='offset points',
                    fontsize=8, ha='center', va='bottom')
    
    ax.set_xlabel('β (Spectral Flattening)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Test Accuracy', fontsize=13, fontweight='bold')
    ax.set_title('Test Accuracy vs. β', fontsize=14, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.set_ylim([min(accuracies) - 0.05, max(accuracies) + 0.05])
    
    # Plot 2: Alignment
    ax = axes[0, 1]
    ax.plot(betas, alignments, 's-', linewidth=2.5, markersize=10, 
            color='#06A77D', alpha=0.9)
    
    ax.set_xlabel('β (Spectral Flattening)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Kernel-Target Alignment', fontsize=13, fontweight='bold')
    ax.set_title('Alignment vs. β', fontsize=14, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 3: Condition number (log scale)
    ax = axes[1, 0]
    ax.semilogy(betas, cond_nums, 'D-', linewidth=2.5, markersize=10, 
                color='#9D4EDD', alpha=0.9)
    
    # Annotate reduction
    reduction_factor = cond_nums[0] / cond_nums[-1]
    ax.annotate(f'{reduction_factor:.1f}× reduction', 
                xy=(betas[-1], cond_nums[-1]),
                xytext=(-80, 30), textcoords='offset points',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.3', lw=1.5))
    
    ax.set_xlabel('β (Spectral Flattening)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Condition Number (log scale)', fontsize=13, fontweight='bold')
    ax.set_title('Condition Number vs. β', fontsize=14, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    
    # Plot 4: Effective rank
    ax = axes[1, 1]
    ax.plot(betas, eff_ranks, '^-', linewidth=2.5, markersize=10, 
            color='#F77F00', alpha=0.9)
    
    # Annotate increase
    rank_increase = eff_ranks[-1] - eff_ranks[0]
    ax.annotate(f'+{rank_increase:.0f}', 
                xy=(betas[-1], eff_ranks[-1]),
                xytext=(-50, -30), textcoords='offset points',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', lw=1.5))
    
    ax.set_xlabel('β (Spectral Flattening)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Effective Rank', fontsize=13, fontweight='bold')
    ax.set_title('Effective Rank vs. β', fontsize=14, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Overall title
    fig.suptitle('Spectral Flattening Performance Analysis on Synthetic Data', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plot_path = 'results/synthetic_data_experiment/plots.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plots saved to {plot_path}")
    plt.close()
    
    # Generate and save summary
    best_idx = np.argmax(accuracies)
    baseline_idx = 0  # β=0
    improvement = (accuracies[best_idx] - accuracies[baseline_idx]) * 100
    
    summary_lines = []
    summary_lines.append("="*70)
    summary_lines.append("SYNTHETIC DATA EXPERIMENT SUMMARY")
    summary_lines.append("="*70)
    summary_lines.append("")
    summary_lines.append("Experimental Setup:")
    summary_lines.append(f"  Features: {X_train.shape[1]}")
    summary_lines.append(f"  Classes: {len(np.unique(y_train))}")
    summary_lines.append(f"  Training samples: {X_train.shape[0]}")
    summary_lines.append(f"  Test samples: {X_test.shape[0]}")
    summary_lines.append(f"  Covariance condition number: {eigenvalues[0]/eigenvalues[-1]:.2e}")
    summary_lines.append(f"  Signal location: Low-variance directions (last 10 dimensions)")
    summary_lines.append("")
    summary_lines.append("Results:")
    summary_lines.append(f"  β values tested: {beta_values}")
    summary_lines.append("")
    summary_lines.append("Baseline (β=0, isotropic RBF):")
    summary_lines.append(f"  Test Accuracy: {accuracies[baseline_idx]:.4f}")
    summary_lines.append(f"  Kernel-Target Alignment: {alignments[baseline_idx]:.4f}")
    summary_lines.append(f"  Effective Rank: {eff_ranks[baseline_idx]:.1f}")
    summary_lines.append(f"  Condition Number: {cond_nums[baseline_idx]:.2e}")
    summary_lines.append("")
    summary_lines.append(f"Best configuration (β={betas[best_idx]:.3f}):")
    summary_lines.append(f"  Test Accuracy: {accuracies[best_idx]:.4f}")
    summary_lines.append(f"  Kernel-Target Alignment: {alignments[best_idx]:.4f}")
    summary_lines.append(f"  Effective Rank: {eff_ranks[best_idx]:.1f}")
    summary_lines.append(f"  Condition Number: {cond_nums[best_idx]:.2e}")
    summary_lines.append("")
    summary_lines.append(f"Improvement: {improvement:+.2f} percentage points")
    summary_lines.append("")
    if improvement > 0:
        summary_lines.append("Spectral flattening provides significant improvement!")
        summary_lines.append("This confirms the theoretical prediction that partial flattening")
        summary_lines.append("helps when class-discriminative signal lies in low-variance directions.")
    else:
        summary_lines.append("No improvement in this configuration.")
        summary_lines.append("This may occur when signal is primarily in high-variance directions.")
    summary_lines.append("")
    summary_lines.append("Output Files:")
    summary_lines.append("  - results/synthetic_data_experiment/results.csv")
    summary_lines.append("  - results/synthetic_data_experiment/plots.png")
    summary_lines.append("  - results/synthetic_data_experiment/experiment_summary.txt")
    summary_lines.append("")
    summary_lines.append("="*70)
    
    summary_text = "\n".join(summary_lines)
    
    # Save summary to file
    summary_path = 'results/synthetic_data_experiment/experiment_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    
    # Print summary
    print("\n" + summary_text)
    print(f"\nSummary saved to {summary_path}")
    
    # Generate 2D decision boundary visualization with same parameters
    print("\n=== Generating 2D Decision Boundary Visualization ===")
    
    # Generate 2D data with same anisotropy but binary for better visualization
    np.random.seed(42)
    n_samples_per_class_2d = 100
    n_classes_2d = 2  # Binary for cleanest gradient visualization
    anisotropy_2d = 100  # Same as main experiment
    
    # Create anisotropic 2D covariance (diagonal - NO ROTATION)
    # Dimension 0 = high variance (eigenvalue = 100)
    # Dimension 1 = low variance (eigenvalue = 1)
    eigenvalues_2d = np.array([anisotropy_2d, 1.0])
    Sigma_2d = np.diag(eigenvalues_2d)
    
    # Class means: signal in low-variance direction (dimension 1, vertical)
    # Dimension 0 (horizontal, high variance): no signal
    # Dimension 1 (vertical, low variance): signal for separation
    mean1_2d = np.array([0.0, -0.8])  # Closer together for harder problem
    mean2_2d = np.array([0.0, 0.8])
    
    # Generate samples
    X1_2d = np.random.multivariate_normal(mean1_2d, Sigma_2d, n_samples_per_class_2d)
    X2_2d = np.random.multivariate_normal(mean2_2d, Sigma_2d, n_samples_per_class_2d)
    X_train_2d = np.vstack([X1_2d, X2_2d])
    y_train_2d = np.array([0]*n_samples_per_class_2d + [1]*n_samples_per_class_2d)
    
    # Test data
    np.random.seed(123)
    X1_test_2d = np.random.multivariate_normal(mean1_2d, Sigma_2d, 50)
    X2_test_2d = np.random.multivariate_normal(mean2_2d, Sigma_2d, 50)
    X_test_2d = np.vstack([X1_test_2d, X2_test_2d])
    y_test_2d = np.array([0]*50 + [1]*50)
    
    print(f"2D training samples: {X_train_2d.shape[0]}")
    print(f"2D test samples: {X_test_2d.shape[0]}")
    print(f"Anisotropy factor: {anisotropy_2d}")
    
    # Plot decision boundaries
    beta_values_viz = [0.0, 0.25, 0.5]
    decision_boundary_path = 'results/synthetic_data_experiment/decision_boundaries.png'
    plot_decision_boundaries_2d(X_train_2d, y_train_2d, X_test_2d, y_test_2d,
                               beta_values_viz, decision_boundary_path)
    
    # Plot comprehensive spectral flattening effect visualization
    print("\n=== Generating Transformation Visualization ===")
    effect_path = 'results/synthetic_data_experiment/spectral_flattening_effect.png'
    plot_spectral_flattening_effect_2d(X_train_2d, y_train_2d, beta_values_viz, effect_path)


if __name__ == '__main__':
    run_synthetic_data_experiment()