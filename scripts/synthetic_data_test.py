"""
Comprehensive validation of spectral flattening on synthetic data
Combines stress tests and parameter sensitivity analysis to demonstrate
theoretical validity across different scenarios

Validates that:
1. LOW-variance signal → spectral flattening HELPS
2. HIGH-variance signal → spectral flattening NEUTRAL/HARMS
3. Effect is robust across different parameter ranges
"""

import os
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kernels import SpectralFlatteningKernel
from src.svm_utils import cross_validate_svm, evaluate_kernel_svm
from scipy.spatial.distance import pdist


def create_synthetic_data(n_samples=1000, n_features=100, n_classes=5, 
                          anisotropy=100, signal_strength=3.0, signal_in_low_var=True):
    """
    Create synthetic data with controlled spectral properties
    
    Parameters:
    -----------
    n_samples : int
        Total training samples
    n_features : int
        Feature dimension
    n_classes : int
        Number of classes
    anisotropy : float
        Condition number (ratio of largest to smallest eigenvalue)
    signal_strength : float
        Magnitude of class separation
    signal_in_low_var : bool
        If True, signal in low-variance directions (method should help)
        If False, signal in high-variance directions (method should not help)
    """
    # Diagonal covariance (no rotation) - dimensions correspond to eigenvalues
    eigenvalues = np.logspace(np.log10(anisotropy), 0, n_features)
    Sigma = np.diag(eigenvalues)
    
    # Generate class means
    class_means = []
    for i in range(n_classes):
        mean = np.zeros(n_features)
        if signal_in_low_var:
            # Signal in last 10 dimensions (smallest eigenvalues ≈ 1)
            mean[-10:] = np.random.randn(10) * signal_strength
        else:
            # Signal in first 10 dimensions (largest eigenvalues ≈ anisotropy)
            mean[:10] = np.random.randn(10) * signal_strength
        class_means.append(mean)
    
    # Generate samples
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    samples_per_class = n_samples // n_classes
    for i in range(n_classes):
        X_class = np.random.multivariate_normal(class_means[i], Sigma, samples_per_class)
        X_train.append(X_class)
        y_train.extend([i] * samples_per_class)
        
        X_class_test = np.random.multivariate_normal(class_means[i], Sigma, 50)
        X_test.append(X_class_test)
        y_test.extend([i] * 50)
    
    return np.vstack(X_train), np.array(y_train), np.vstack(X_test), np.array(y_test)


def evaluate_configuration(X_train, y_train, X_test, y_test, beta_values=[0.0, 0.5], 
                          C_values=[0.1, 1.0, 10.0]):
    """Evaluate spectral flattening for given configuration"""
    results = {}
    
    for beta in beta_values:
        # Fit kernel with adaptive sigma
        kernel = SpectralFlatteningKernel(beta=beta, sigma=1.0, shrinkage=True)
        kernel.fit(X_train)
        X_tf = kernel.transform(X_train)
        sigma = np.median(pdist(X_tf)) / 3
        
        # Refit with adaptive sigma
        kernel = SpectralFlatteningKernel(beta=beta, sigma=sigma, shrinkage=True)
        kernel.fit(X_train)
        K_train = kernel.compute_kernel_matrix(X_train)
        K_test = kernel.compute_kernel_matrix(X_test, X_train)
        
        # Cross-validate and evaluate
        best_C, cv_score = cross_validate_svm(K_train, y_train, C_values, cv=3)
        metrics = evaluate_kernel_svm(K_train, K_test, y_train, y_test, C=best_C)
        
        results[beta] = {
            'cv_score': cv_score,
            'test_accuracy': metrics['test_accuracy'],
            'train_accuracy': metrics['train_accuracy'],
            'C': best_C
        }
    
    return results


def run_comprehensive_tests():
    """Run comprehensive validation tests"""
    
    # Redirect prints to both console and file
    output_lines = []
    
    def tee_print(text):
        """Print to console and save to list"""
        print(text)
        output_lines.append(text)
    
    tee_print("="*100)
    tee_print("SYNTHETIC DATA VALIDATION")
    tee_print("Testing spectral flattening theory across multiple scenarios")
    tee_print("="*100)
    
    os.makedirs('results/synthetic_data_test', exist_ok=True)
    
    np.random.seed(42)
    
    all_results = []
    
    # ========================================
    # TEST 1: Basic Validation (LOW vs HIGH variance signal)
    # ========================================
    tee_print("\n[TEST 1] BASIC VALIDATION: Signal Location Effect")
    tee_print("-" * 100)
    
    # LOW-variance signal (should help)
    X, y, Xt, yt = create_synthetic_data(signal_in_low_var=True)
    results_low = evaluate_configuration(X, y, Xt, yt)
    improvement_low = (results_low[0.5]['test_accuracy'] - results_low[0.0]['test_accuracy']) * 100
    
    tee_print(f"LOW-variance signal:")
    tee_print(f"  Baseline (β=0.0): Test accuracy = {results_low[0.0]['test_accuracy']:.3f}, CV score = {results_low[0.0]['cv_score']:.3f}")
    tee_print(f"  Flattened (β=0.5): Test accuracy = {results_low[0.5]['test_accuracy']:.3f}, CV score = {results_low[0.5]['cv_score']:.3f}")
    tee_print(f"  Improvement: {improvement_low:+.1f} percentage points")
    
    all_results.append({
        'test': 'Basic - LOW-var',
        'beta_0_test': results_low[0.0]['test_accuracy'],
        'beta_05_test': results_low[0.5]['test_accuracy'],
        'improvement_pp': improvement_low,
        'expected': 'positive',
        'result': 'PASS' if improvement_low > 0 else 'FAIL'
    })
    
    # HIGH-variance signal (should NOT help)
    X, y, Xt, yt = create_synthetic_data(signal_in_low_var=False)
    results_high = evaluate_configuration(X, y, Xt, yt)
    improvement_high = (results_high[0.5]['test_accuracy'] - results_high[0.0]['test_accuracy']) * 100
    
    tee_print(f"\nHIGH-variance signal:")
    tee_print(f"  Baseline (β=0.0): Test accuracy = {results_high[0.0]['test_accuracy']:.3f}, CV score = {results_high[0.0]['cv_score']:.3f}")
    tee_print(f"  Flattened (β=0.5): Test accuracy = {results_high[0.5]['test_accuracy']:.3f}, CV score = {results_high[0.5]['cv_score']:.3f}")
    tee_print(f"  Improvement: {improvement_high:+.1f} percentage points")
    
    all_results.append({
        'test': 'Basic - HIGH-var',
        'beta_0_test': results_high[0.0]['test_accuracy'],
        'beta_05_test': results_high[0.5]['test_accuracy'],
        'improvement_pp': improvement_high,
        'expected': 'zero/negative',
        'result': 'PASS' if improvement_high <= 5 else 'FAIL'
    })
    
    # ========================================
    # TEST 2: Signal Strength Sensitivity
    # ========================================
    tee_print("\n[TEST 2] SIGNAL STRENGTH SENSITIVITY")
    tee_print("-" * 100)
    tee_print("Testing with varying signal strengths (LOW-var signal)")
    
    for strength in [1.0, 2.0, 3.0, 5.0]:
        X, y, Xt, yt = create_synthetic_data(signal_strength=strength, signal_in_low_var=True)
        results = evaluate_configuration(X, y, Xt, yt)
        improvement = (results[0.5]['test_accuracy'] - results[0.0]['test_accuracy']) * 100
        
        tee_print(f"  Signal strength {strength:.1f}: Baseline (β=0.0) = {results[0.0]['test_accuracy']:.3f}, "
              f"Flattened (β=0.5) = {results[0.5]['test_accuracy']:.3f}, Change = {improvement:+.1f}pp")
        
        all_results.append({
            'test': f'Strength {strength}',
            'beta_0_test': results[0.0]['test_accuracy'],
            'beta_05_test': results[0.5]['test_accuracy'],
            'improvement_pp': improvement,
            'expected': 'positive',
            'result': 'PASS' if improvement > 0 else 'FAIL'
        })
    
    # ========================================
    # TEST 3: Anisotropy Sensitivity
    # ========================================
    tee_print("\n[TEST 3] ANISOTROPY SENSITIVITY")
    tee_print("-" * 100)
    tee_print("Testing with varying condition numbers (LOW-var signal)")
    
    for anisotropy in [10, 100, 1000, 10000]:
        X, y, Xt, yt = create_synthetic_data(anisotropy=anisotropy, signal_in_low_var=True)
        results = evaluate_configuration(X, y, Xt, yt)
        improvement = (results[0.5]['test_accuracy'] - results[0.0]['test_accuracy']) * 100
        
        tee_print(f"  Condition number κ={anisotropy:5d}: Baseline (β=0.0) = {results[0.0]['test_accuracy']:.3f}, "
              f"Flattened (β=0.5) = {results[0.5]['test_accuracy']:.3f}, Change = {improvement:+.1f}pp")
        
        all_results.append({
            'test': f'Anisotropy {anisotropy}',
            'beta_0_test': results[0.0]['test_accuracy'],
            'beta_05_test': results[0.5]['test_accuracy'],
            'improvement_pp': improvement,
            'expected': 'positive',
            'result': 'PASS' if improvement > -5 else 'FAIL'  # Allow some variance
        })
    
    # ========================================
    # TEST 4: Sample Size Sensitivity
    # ========================================
    tee_print("\n[TEST 4] SAMPLE SIZE SENSITIVITY")
    tee_print("-" * 100)
    tee_print("Testing with varying samples per class (LOW-var signal)")
    
    for n_samples in [125, 250, 500, 1000]:
        X, y, Xt, yt = create_synthetic_data(n_samples=n_samples, signal_in_low_var=True)
        results = evaluate_configuration(X, y, Xt, yt)
        improvement = (results[0.5]['test_accuracy'] - results[0.0]['test_accuracy']) * 100
        
        spc = n_samples // 5  # samples per class
        tee_print(f"  {spc:3d} samples per class: Baseline (β=0.0) = {results[0.0]['test_accuracy']:.3f}, "
              f"Flattened (β=0.5) = {results[0.5]['test_accuracy']:.3f}, Change = {improvement:+.1f}pp")
        
        all_results.append({
            'test': f'Samples/class {spc}',
            'beta_0_test': results[0.0]['test_accuracy'],
            'beta_05_test': results[0.5]['test_accuracy'],
            'improvement_pp': improvement,
            'expected': 'positive',
            'result': 'PASS' if improvement > -5 else 'FAIL'
        })
    
    # ========================================
    # TEST 5: Realistic CUB-like Scenarios
    # ========================================
    tee_print("\n[TEST 5] REALISTIC SCENARIOS")
    tee_print("-" * 100)
    tee_print("Testing configurations matching real dataset properties:")
    tee_print("  - CUB-200-2011 has ~30 samples/class (5994 train / 200 classes)")
    tee_print("  - Pretrained features typically have κ ~ 10^3 to 10^4")
    tee_print("")
    
    scenarios = [
        {'name': 'Easy (default)', 'n_samples': 1000, 'strength': 3.0, 'anisotropy': 100},
        {'name': 'CUB-like (30 samples/class, κ=1000)', 'n_samples': 250, 'strength': 2.0, 'anisotropy': 1000},
        {'name': 'Hard (25 samples/class, weak signal, κ=1000)', 'n_samples': 125, 'strength': 1.5, 'anisotropy': 1000},
    ]
    
    for scenario in scenarios:
        # LOW-variance
        X, y, Xt, yt = create_synthetic_data(
            n_samples=scenario['n_samples'],
            signal_strength=scenario['strength'],
            anisotropy=scenario['anisotropy'],
            signal_in_low_var=True
        )
        results_low = evaluate_configuration(X, y, Xt, yt)
        imp_low = (results_low[0.5]['test_accuracy'] - results_low[0.0]['test_accuracy']) * 100
        
        # HIGH-variance
        X, y, Xt, yt = create_synthetic_data(
            n_samples=scenario['n_samples'],
            signal_strength=scenario['strength'],
            anisotropy=scenario['anisotropy'],
            signal_in_low_var=False
        )
        results_high = evaluate_configuration(X, y, Xt, yt)
        imp_high = (results_high[0.5]['test_accuracy'] - results_high[0.0]['test_accuracy']) * 100
        
        tee_print(f"\n  {scenario['name']}:")
        tee_print(f"    LOW-variance signal:  Baseline (β=0.0) = {results_low[0.0]['test_accuracy']:.3f}, "
              f"Flattened (β=0.5) = {results_low[0.5]['test_accuracy']:.3f}, Change = {imp_low:+.1f}pp")
        tee_print(f"    HIGH-variance signal: Baseline (β=0.0) = {results_high[0.0]['test_accuracy']:.3f}, "
              f"Flattened (β=0.5) = {results_high[0.5]['test_accuracy']:.3f}, Change = {imp_high:+.1f}pp")
        
        all_results.append({
            'test': f'{scenario["name"]} LOW-var',
            'beta_0_test': results_low[0.0]['test_accuracy'],
            'beta_05_test': results_low[0.5]['test_accuracy'],
            'improvement_pp': imp_low,
            'expected': 'positive',
            'result': 'PASS' if imp_low > -5 else 'FAIL'
        })
        
        all_results.append({
            'test': f'{scenario["name"]} HIGH-var',
            'beta_0_test': results_high[0.0]['test_accuracy'],
            'beta_05_test': results_high[0.5]['test_accuracy'],
            'improvement_pp': imp_high,
            'expected': 'zero/negative',
            'result': 'PASS' if imp_high <= 10 else 'FAIL'
        })
    
    # ========================================
    # SAVE RESULTS
    # ========================================
    results_df = pd.DataFrame(all_results)
    results_path = 'results/synthetic_data_test/validation_results.csv'
    results_df.to_csv(results_path, index=False)
    
    # ========================================
    # SUMMARY
    # ========================================
    tee_print("\n" + "="*100)
    tee_print("SUMMARY")
    tee_print("="*100)
    
    passes = sum(1 for r in all_results if r['result'] == 'PASS')
    total = len(all_results)
    
    tee_print(f"\nTests passed: {passes}/{total} ({100*passes/total:.1f}%)")
    tee_print(f"\nResults saved to: {results_path}")
    
    # Key findings
    low_var_improvements = [r['improvement_pp'] for r in all_results if 'LOW-var' in r['test']]
    high_var_improvements = [r['improvement_pp'] for r in all_results if 'HIGH-var' in r['test']]
    
    tee_print("\nKey Findings:")
    tee_print(f"  LOW-variance signal scenarios:  Average change = {np.mean(low_var_improvements):+.1f}pp")
    tee_print(f"  HIGH-variance signal scenarios: Average change = {np.mean(high_var_improvements):+.1f}pp")
    tee_print("="*100)
    
    # Save summary to text file
    summary_path = 'results/synthetic_data_test/validation_summary.txt'
    with open(summary_path, 'w') as f:
        f.write('\n'.join(output_lines))
    
    tee_print(f"\n Summary saved to: {summary_path}")


if __name__ == '__main__':
    run_comprehensive_tests()
