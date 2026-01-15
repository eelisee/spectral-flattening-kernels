"""
Systematic test of spectral flattening under different covariance structures
Tests the central hypothesis: Δ = U α must be constructed in eigenspace

Tests 4 covariance types × 3 signal profiles = 12 scenarios
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import pandas as pd
from scipy.linalg import toeplitz

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kernels import SpectralFlatteningKernel
from src.svm_utils import evaluate_kernel_svm, cross_validate_svm


def create_covariance_matrix(n_features, cov_type, anisotropy=100):
    """
    Create different types of realistic covariance structures
    
    Args:
        n_features: Dimension
        cov_type: 'random', 'toeplitz', 'block', 'lowrank_noise'
        anisotropy: Condition number
        
    Returns:
        Sigma, U, Lambda (covariance, eigenvectors, eigenvalues)
    """
    # Generate eigenvalue spectrum (power-law decay)
    eigenvalues = np.logspace(np.log10(anisotropy), 0, n_features)
    Lambda = eigenvalues
    
    if cov_type == 'random':
        # Random orthonormal basis (maximally incoherent)
        U, _ = np.linalg.qr(np.random.randn(n_features, n_features))
        Sigma = U @ np.diag(eigenvalues) @ U.T
        
    elif cov_type == 'toeplitz':
        # AR(1)-like structure with local correlations
        rho = 0.7
        row = rho ** np.arange(n_features)
        Sigma_base = toeplitz(row)
        # Rescale to desired spectrum
        eigvals_base, eigvecs_base = np.linalg.eigh(Sigma_base)
        idx = np.argsort(eigvals_base)[::-1]
        eigvals_base = eigvals_base[idx]
        eigvecs_base = eigvecs_base[:, idx]
        U = eigvecs_base
        Sigma = U @ np.diag(eigenvalues) @ U.T
        
    elif cov_type == 'block':
        # Block-diagonal structure (feature groups)
        n_blocks = 4
        block_size = n_features // n_blocks
        blocks = []
        U_blocks = []
        for i in range(n_blocks):
            # Each block has its own structure
            block_eigenvalues = eigenvalues[i*block_size:(i+1)*block_size]
            U_block, _ = np.linalg.qr(np.random.randn(block_size, block_size))
            block = U_block @ np.diag(block_eigenvalues) @ U_block.T
            blocks.append(block)
            U_blocks.append(U_block)
        Sigma = np.zeros((n_features, n_features))
        U = np.zeros((n_features, n_features))
        for i, (block, U_block) in enumerate(zip(blocks, U_blocks)):
            start = i * block_size
            end = start + block_size
            Sigma[start:end, start:end] = block
            U[start:end, start:end] = U_block
        # Recompute eigendecomposition for consistency
        eigenvalues, U = np.linalg.eigh(Sigma)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        U = U[:, idx]
        Lambda = eigenvalues
        
    elif cov_type == 'lowrank_noise':
        # Strong signal subspace + noise (realistic for deep features)
        k = n_features // 4  # Rank of signal subspace
        noise_level = 0.1
        
        # Signal eigenvalues (strong)
        signal_eigenvalues = eigenvalues[:k]
        # Noise eigenvalues (flat)
        noise_eigenvalues = np.ones(n_features - k) * noise_level
        
        Lambda = np.concatenate([signal_eigenvalues, noise_eigenvalues])
        U, _ = np.linalg.qr(np.random.randn(n_features, n_features))
        Sigma = U @ np.diag(Lambda) @ U.T
        
    else:
        raise ValueError(f"Unknown covariance type: {cov_type}")
    
    return Sigma, U, Lambda


def create_signal_in_eigenspace(U, Lambda, signal_type, signal_strength=1.0, k=10):
    """
    Create signal Δ = U α in eigenspace
    
    Args:
        U: Eigenvectors
        Lambda: Eigenvalues
        signal_type: 'low_var', 'high_var', 'power_law'
        signal_strength: Overall signal magnitude
        k: Number of non-zero components
        
    Returns:
        Delta (signal vector in original space)
    """
    d = len(Lambda)
    alpha = np.zeros(d)
    
    if signal_type == 'low_var':
        # Signal in low-variance eigenvectors (last k)
        alpha[-k:] = np.random.randn(k) * signal_strength
        
    elif signal_type == 'high_var':
        # Signal in high-variance eigenvectors (first k)
        alpha[:k] = np.random.randn(k) * signal_strength
        
    elif signal_type == 'power_law':
        # Signal follows power law: α_i ∝ λ_i^(-0.5)
        # This creates signal anti-correlated with variance
        alpha = np.random.randn(d) * signal_strength * (Lambda ** (-0.5))
        alpha /= np.linalg.norm(alpha)  # Normalize
        alpha *= signal_strength * np.sqrt(k)  # Rescale to comparable magnitude
        
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")
    
    # Project to original space
    Delta = U @ alpha
    
    return Delta


def generate_synthetic_data(Sigma, Delta, n_samples=1000, n_classes=5):
    """
    Generate synthetic data with given covariance and signal
    
    Args:
        Sigma: Covariance matrix
        Delta: Signal vector (class separation)
        n_samples: Total samples
        n_classes: Number of classes
        
    Returns:
        X_train, y_train, X_test, y_test
    """
    d = Sigma.shape[0]
    
    # Create class means along signal direction
    class_means = []
    for i in range(n_classes):
        # Linearly space classes along Delta direction
        t = (i - (n_classes - 1) / 2) / (n_classes - 1)
        mean = t * Delta
        class_means.append(mean)
    
    # Generate samples
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    for i, mean in enumerate(class_means):
        # Training samples
        X_class = np.random.multivariate_normal(mean, Sigma, n_samples // n_classes)
        X_train.append(X_class)
        y_train.extend([i] * (n_samples // n_classes))
        
        # Test samples
        X_class_test = np.random.multivariate_normal(mean, Sigma, 50)
        X_test.append(X_class_test)
        y_test.extend([i] * 50)
    
    X_train = np.vstack(X_train)
    X_test = np.vstack(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    return X_train, y_train, X_test, y_test


def run_covariance_structure_test():
    """
    Main test function: 4 covariance types × 3 signal profiles
    """
    print("="*80)
    print("COVARIANCE STRUCTURE TEST")
    print("Testing spectral flattening under realistic covariance structures")
    print("="*80)
    
    # Setup
    os.makedirs('results/covariance_structure_test/', exist_ok=True)
    np.random.seed(42)
    
    # Parameters
    n_features = 100
    n_samples = 1000
    n_classes = 5
    anisotropy = 100
    
    cov_types = ['random', 'toeplitz', 'block', 'lowrank_noise']
    signal_types = ['low_var', 'high_var', 'power_law']
    beta_values = [0.0, 0.125, 0.25, 0.375, 0.5]
    C_values = [0.1, 1.0, 10.0]
    
    all_results = []
    
    for cov_type in cov_types:
        print(f"\n{'='*80}")
        print(f"COVARIANCE TYPE: {cov_type.upper()}")
        print(f"{'='*80}")
        
        # Create covariance structure
        Sigma, U, Lambda = create_covariance_matrix(n_features, cov_type, anisotropy)
        print(f"Condition number: {Lambda[0] / Lambda[-1]:.2e}")
        
        for signal_type in signal_types:
            print(f"\n  Signal profile: {signal_type}")
            
            # Create signal in eigenspace
            Delta = create_signal_in_eigenspace(U, Lambda, signal_type, 
                                               signal_strength=1.0, k=10)
            
            # Generate data
            X_train, y_train, X_test, y_test = generate_synthetic_data(
                Sigma, Delta, n_samples, n_classes
            )
            
            # Test different β values
            scenario_results = []
            
            for beta in beta_values:
                # Fit kernel
                kernel = SpectralFlatteningKernel(beta=beta, sigma=1.0, shrinkage=False)
                kernel.fit(X_train)
                
                # Adaptive sigma
                from scipy.spatial.distance import pdist
                X_trans = kernel.transform(X_train)
                sigma_adaptive = np.median(pdist(X_trans, metric='euclidean')) / 3.0
                
                # Refit with adaptive sigma
                kernel = SpectralFlatteningKernel(beta=beta, sigma=sigma_adaptive, shrinkage=False)
                kernel.fit(X_train)
                
                # Compute kernels
                K_train = kernel.compute_kernel_matrix(X_train)
                K_test = kernel.compute_kernel_matrix(X_test, X_train)
                
                # Cross-validate C
                best_C, cv_score = cross_validate_svm(K_train, y_train, C_values, cv=3)
                
                # Test accuracy
                metrics = evaluate_kernel_svm(K_train, K_test, y_train, y_test, C=best_C)
                
                scenario_results.append({
                    'cov_type': cov_type,
                    'signal_type': signal_type,
                    'beta': beta,
                    'sigma': sigma_adaptive,
                    'C': best_C,
                    'cv_score': cv_score,
                    'test_accuracy': metrics['test_accuracy'],
                    'train_accuracy': metrics['train_accuracy']
                })
            
            # Print summary for this scenario
            baseline_acc = scenario_results[0]['test_accuracy']
            best_idx = np.argmax([r['test_accuracy'] for r in scenario_results])
            best_acc = scenario_results[best_idx]['test_accuracy']
            best_beta = scenario_results[best_idx]['beta']
            improvement = (best_acc - baseline_acc) * 100
            
            print(f"    Baseline (β=0.0): {baseline_acc:.4f}")
            print(f"    Best (β={best_beta:.3f}): {best_acc:.4f}")
            print(f"    Change: {improvement:+.2f}pp")
            
            all_results.extend(scenario_results)
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('results/covariance_structure_test/results.csv', index=False)
    print(f"\nResults saved to results/covariance_structure_test/results.csv")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_plots(results_df)
    
    # Generate summary
    generate_summary(results_df)
    
    return results_df


def generate_plots(results_df):
    """Generate comprehensive visualization of results"""
    
    cov_types = results_df['cov_type'].unique()
    signal_types = results_df['signal_type'].unique()
    
    # Plot 1: Performance curves for each scenario
    fig, axes = plt.subplots(len(cov_types), len(signal_types), 
                            figsize=(15, 12), sharex=True, sharey=True)
    
    for i, cov_type in enumerate(cov_types):
        for j, signal_type in enumerate(signal_types):
            ax = axes[i, j]
            
            # Filter data
            mask = (results_df['cov_type'] == cov_type) & \
                   (results_df['signal_type'] == signal_type)
            data = results_df[mask].sort_values('beta')
            
            # Plot
            ax.plot(data['beta'], data['test_accuracy'], 'o-', 
                   linewidth=2, markersize=8, color='blue')
            
            # Formatting
            if i == 0:
                ax.set_title(signal_type.replace('_', ' ').title(), 
                           fontsize=11, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'{cov_type.title()}\nTest Accuracy', 
                            fontsize=10, fontweight='bold')
            if i == len(cov_types) - 1:
                ax.set_xlabel('β', fontsize=10)
            
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
            
            # Highlight baseline
            baseline = data[data['beta'] == 0.0]['test_accuracy'].values[0]
            ax.axhline(y=baseline, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('results/covariance_structure_test/performance_grid.png', 
               dpi=200, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Improvement heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Compute improvements (best β vs baseline)
    improvements = []
    scenario_labels = []
    
    for cov_type in cov_types:
        for signal_type in signal_types:
            mask = (results_df['cov_type'] == cov_type) & \
                   (results_df['signal_type'] == signal_type)
            data = results_df[mask]
            baseline = data[data['beta'] == 0.0]['test_accuracy'].values[0]
            best = data['test_accuracy'].max()
            improvement = (best - baseline) * 100
            improvements.append(improvement)
            scenario_labels.append(f"{cov_type}\n{signal_type.replace('_', ' ')}")
    
    colors = ['red' if x < 0 else 'green' for x in improvements]
    bars = ax.barh(range(len(improvements)), improvements, color=colors, alpha=0.7)
    
    ax.set_yticks(range(len(improvements)))
    ax.set_yticklabels(scenario_labels, fontsize=9)
    ax.set_xlabel('Improvement (percentage points)', fontsize=11, fontweight='bold')
    ax.set_title('Spectral Flattening: Best β vs Baseline (β=0)', 
                fontsize=12, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('results/covariance_structure_test/improvement_heatmap.png', 
               dpi=200, bbox_inches='tight')
    plt.close()
    
    print("Plots saved:")
    print("  - results/covariance_structure_test/performance_grid.png")
    print("  - results/covariance_structure_test/improvement_heatmap.png")


def generate_summary(results_df):
    """Generate text summary of results"""
    
    summary_lines = []
    summary_lines.append("="*80)
    summary_lines.append("COVARIANCE STRUCTURE TEST - SUMMARY")
    summary_lines.append("="*80)
    summary_lines.append("")
    summary_lines.append("Test Setup:")
    summary_lines.append("  Signal construction: Δ = U α (in eigenspace)")
    summary_lines.append("  Covariance types: 4 (random, toeplitz, block, lowrank+noise)")
    summary_lines.append("  Signal profiles: 3 (low-variance, high-variance, power-law)")
    summary_lines.append("  Total scenarios: 12")
    summary_lines.append("")
    summary_lines.append("-"*80)
    summary_lines.append("Results by Scenario:")
    summary_lines.append("-"*80)
    
    cov_types = results_df['cov_type'].unique()
    signal_types = results_df['signal_type'].unique()
    
    for cov_type in cov_types:
        summary_lines.append(f"\n{cov_type.upper()} Covariance:")
        
        for signal_type in signal_types:
            mask = (results_df['cov_type'] == cov_type) & \
                   (results_df['signal_type'] == signal_type)
            data = results_df[mask]
            
            baseline = data[data['beta'] == 0.0]['test_accuracy'].values[0]
            best_idx = data['test_accuracy'].idxmax()
            best = data.loc[best_idx]
            improvement = (best['test_accuracy'] - baseline) * 100
            
            summary_lines.append(f"  {signal_type.replace('_', ' ').title()}:")
            summary_lines.append(f"    Baseline (β=0.0): {baseline:.4f}")
            summary_lines.append(f"    Best (β={best['beta']:.3f}): {best['test_accuracy']:.4f}")
            summary_lines.append(f"    Change: {improvement:+.2f}pp")
    
    summary_lines.append("")
    summary_lines.append("="*80)
    summary_lines.append("KEY FINDINGS:")
    summary_lines.append("="*80)
    
    # Analyze patterns
    summary_lines.append("")
    summary_lines.append("1. LOW-VARIANCE SIGNAL SCENARIOS:")
    low_var_improvements = []
    for cov_type in cov_types:
        mask = (results_df['cov_type'] == cov_type) & \
               (results_df['signal_type'] == 'low_var')
        data = results_df[mask]
        baseline = data[data['beta'] == 0.0]['test_accuracy'].values[0]
        best = data['test_accuracy'].max()
        improvement = (best - baseline) * 100
        low_var_improvements.append(improvement)
        summary_lines.append(f"   {cov_type}: {improvement:+.2f}pp")
    
    summary_lines.append(f"   Average: {np.mean(low_var_improvements):+.2f}pp")
    summary_lines.append("")
    
    summary_lines.append("2. HIGH-VARIANCE SIGNAL SCENARIOS:")
    high_var_improvements = []
    for cov_type in cov_types:
        mask = (results_df['cov_type'] == cov_type) & \
               (results_df['signal_type'] == 'high_var')
        data = results_df[mask]
        baseline = data[data['beta'] == 0.0]['test_accuracy'].values[0]
        best = data['test_accuracy'].max()
        improvement = (best - baseline) * 100
        high_var_improvements.append(improvement)
        summary_lines.append(f"   {cov_type}: {improvement:+.2f}pp")
    
    summary_lines.append(f"   Average: {np.mean(high_var_improvements):+.2f}pp")
    summary_lines.append("")
    
    summary_lines.append("3. POWER-LAW SIGNAL SCENARIOS:")
    power_law_improvements = []
    for cov_type in cov_types:
        mask = (results_df['cov_type'] == cov_type) & \
               (results_df['signal_type'] == 'power_law')
        data = results_df[mask]
        baseline = data[data['beta'] == 0.0]['test_accuracy'].values[0]
        best = data['test_accuracy'].max()
        improvement = (best - baseline) * 100
        power_law_improvements.append(improvement)
        summary_lines.append(f"   {cov_type}: {improvement:+.2f}pp")
    
    summary_lines.append(f"   Average: {np.mean(power_law_improvements):+.2f}pp")
    summary_lines.append("")
    summary_lines.append("="*80)
    summary_lines.append("Output Files:")
    summary_lines.append("  - results/covariance_structure_test/results.csv")
    summary_lines.append("  - results/covariance_structure_test/performance_grid.png")
    summary_lines.append("  - results/covariance_structure_test/improvement_heatmap.png")
    summary_lines.append("  - results/covariance_structure_test/summary.txt")
    summary_lines.append("")
    summary_lines.append("="*80)
    
    summary_text = "\n".join(summary_lines)
    
    # Save summary
    with open('results/covariance_structure_test/summary.txt', 'w') as f:
        f.write(summary_text)
    
    # Print summary
    print("\n" + summary_text)


if __name__ == '__main__':
    run_covariance_structure_test()
