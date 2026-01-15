"""
Test grid_search_kernel_params() on synthetic data
Using same data as synthetic_data_experiment.py for reproducibility
"""

import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kernels import SpectralFlatteningKernel
from src.svm_utils import grid_search_kernel_params, evaluate_kernel_svm


def create_synthetic_features(n_samples=1000, n_features=100, n_classes=5, 
                              anisotropy_factor=100, signal_in_low_var=True):
    """
    Create synthetic features with controlled spectral structure as in synthetic_data_experiment.py
    """
    print("\n=== Generating Synthetic Data ===")
    
    # Create anisotropic covariance (diagonal - NO ROTATION)
    # This ensures dimension i corresponds directly to eigenvalue i
    eigenvalues = np.logspace(np.log10(anisotropy_factor), 0, n_features)
    Sigma = np.diag(eigenvalues)  # No rotation! Dimensions = eigenvectors
    
    print(f"Feature dimension: {n_features}")
    print(f"Number of classes: {n_classes}")
    print(f"Covariance condition number: {eigenvalues[0]/eigenvalues[-1]:.2e}")
    
    # Generate class means
    class_means = []
    for i in range(n_classes):
        if signal_in_low_var:
            # Signal in low-variance directions (should benefit from flattening)
            mean = np.zeros(n_features)
            mean[-10:] = np.random.randn(10) 
        else:
            # Signal in high-variance directions
            mean = np.zeros(n_features)
            mean[:10] = np.random.randn(10)
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


def test_grid_search():
    """Test grid search on synthetic data"""
    
    print("="*70)
    print("TESTING GRID_SEARCH_KERNEL_PARAMS ON SYNTHETIC DATA")
    print("="*70)
    
    # Generate EXACT SAME data as in synthetic_data_experiment.py
    np.random.seed(42)  # Same seed for reproducibility
    X_train, y_train, X_test, y_test, Sigma, eigenvalues = create_synthetic_features(
        n_samples=1000, 
        n_features=100, 
        n_classes=5, 
        anisotropy_factor=100, 
        signal_in_low_var=True
    )
    
    # Same parameter ranges as in synthetic_data_experiment.py
    beta_values = [0.0, 0.125, 0.25, 0.375, 0.5]
    C_values = [0.1, 1.0, 10.0, 100.0]
    
    # Run grid search with adaptive sigma
    print("\n" + "="*70)
    print("RUNNING GRID SEARCH")
    print("="*70)
    
    grid_results = grid_search_kernel_params(
        X_train=X_train,
        y_train=y_train,
        beta_values=beta_values,
        C_values=C_values,
        cv=3,  # 3-fold CV (faster for testing)
        use_adaptive_sigma=True,
        sigma_scale_factor=3.0
    )
    
    # Extract best parameters
    best_params = grid_results['best_params']
    best_score = grid_results['best_score']
    
    print("\n" + "="*70)
    print("EVALUATING BEST CONFIGURATION ON TEST SET")
    print("="*70)
    
    # Fit kernel with best parameters and evaluate on test set
    kernel = SpectralFlatteningKernel(
        beta=best_params['beta'], 
        sigma=best_params['sigma'], 
        shrinkage=True
    )
    kernel.fit(X_train)
    
    K_train = kernel.compute_kernel_matrix(X_train)
    K_test = kernel.compute_kernel_matrix(X_test, X_train)
    
    test_results = evaluate_kernel_svm(
        K_train=K_train,
        K_test=K_test,
        y_train=y_train,
        y_test=y_test,
        C=best_params['C']
    )
    
    print(f"\nBest Configuration:")
    print(f"  β = {best_params['beta']:.3f}")
    print(f"  σ = {best_params['sigma']:.2f}")
    print(f"  C = {best_params['C']:.2f}")
    print(f"\nCross-Validation Score: {best_score:.4f}")
    print(f"\nTest Set Results:")
    print(f"  Test Accuracy:      {test_results['test_accuracy']:.4f}")
    print(f"  Train Accuracy:     {test_results['train_accuracy']:.4f}")
    
    # Show all results for comparison
    print("\n" + "="*70)
    print("ALL TESTED CONFIGURATIONS")
    print("="*70)
    print(f"{'β':<8} {'σ':<10} {'C':<10} {'CV Score':<12}")
    print("-"*70)
    for result in grid_results['all_results']:
        print(f"{result['beta']:<8.3f} {result['sigma']:<10.2f} {result['C']:<10.2f} {result['cv_score']:<12.4f}")
    
    print("\n" + "="*70)
    print("COMPARISON WITH BASELINE (β=0.0)")
    print("="*70)
    
    # Find baseline result (β=0.0)
    baseline = [r for r in grid_results['all_results'] if r['beta'] == 0.0][0]
    improvement = (best_score - baseline['cv_score']) * 100
    
    print(f"Baseline (β=0.0): CV Score = {baseline['cv_score']:.4f}")
    print(f"Best (β={best_params['beta']:.3f}): CV Score = {best_score:.4f}")
    print(f"Improvement: +{improvement:.2f} percentage points")
    
    return grid_results, test_results


if __name__ == "__main__":
    results = test_grid_search()
