"""
Real-Data Evaluation Script
Evaluates spectral flattening on benchmark datasets with deep features
Uses grid search to find optimal (β, σ, C) per architecture

Supported datasets:
- CUB-200-2011 (200 bird species)
- Stanford Cars (196 car models)

Usage:
    python scripts/evaluate_real_data.py --dataset cub --model resnet50
    python scripts/evaluate_real_data.py --dataset stanford_cars --model vit_b_16
    python scripts/evaluate_real_data.py --dataset cub --all
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy.spatial.distance import pdist

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kernels import (SpectralFlatteningKernel, compute_kernel_target_alignment,
                         compute_effective_rank)
from src.svm_utils import grid_search_kernel_params, evaluate_kernel_svm
from src.visualization import plot_decision_boundaries_2d


def evaluate_model(dataset_name, model_name, beta_values=None, C_values=None, cv_folds=3):
    """
    Evaluate spectral flattening on real data for one model architecture
    
    Args:
        dataset_name: 'cub' or 'stanford_cars'
        model_name: 'resnet50', 'vit_b_16', or 'convnext_base'
        beta_values: List of β values to test
        C_values: List of C values for grid search
        cv_folds: Number of cross-validation folds
    """
    
    dataset_display = {'cub': 'CUB-200-2011', 'stanford_cars': 'Stanford_Cars'}[dataset_name]
    
    print("="*80)
    print(f"SPECTRAL FLATTENING EVALUATION")
    print(f"Dataset: {dataset_display}")
    print(f"Model: {model_name.upper()}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Setup
    output_dir = f'results/real_data/{dataset_display}/{model_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Default parameters (matching synthetic_data_experiment.py)
    if beta_values is None:
        beta_values = [0.0, 0.025, 0.05, 0.075, 0.10, 0.125]
    if C_values is None:
        # Use much smaller C values to prevent overfitting
        # Stanford Cars has 196 classes with ~40 samples per class
        # → Need strong regularization
        C_values = [0.01, 0.05, 0.1, 0.5, 1.0]
    
    # Load features
    print(f"\n1. Loading {model_name} features for {dataset_display}...")
    data_train = np.load(f'results/features/{dataset_name}/{model_name}_train.npz')
    data_test = np.load(f'results/features/{dataset_name}/{model_name}_test.npz')
    
    X_train = data_train['features']
    y_train = data_train['labels']
    X_test = data_test['features']
    y_test = data_test['labels']
    
    print(f"   Train: {X_train.shape}")
    print(f"   Test:  {X_test.shape}")
    print(f"   Classes: {len(np.unique(y_train))}")
    
    # Spectral analysis
    print(f"\n2. Spectral Analysis...")
    from sklearn.covariance import LedoitWolf
    from scipy.linalg import eigh
    
    lw = LedoitWolf()
    Sigma = lw.fit(X_train).covariance_
    eigenvalues, _ = eigh(Sigma)
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    condition_number = eigenvalues[0] / eigenvalues[-1]
    effective_rank_orig = np.sum(eigenvalues) ** 2 / np.sum(eigenvalues ** 2)
    cumvar = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    
    print(f"   Condition number: {condition_number:.2e}")
    print(f"   Effective rank: {effective_rank_orig:.1f} / {len(eigenvalues)}")
    print(f"   Top 10 PCs: {cumvar[9]:.2%} variance")
    print(f"   Top 50 PCs: {cumvar[49]:.2%} variance")
    print(f"   Top 100 PCs: {cumvar[99]:.2%} variance")
    
    # Grid search for optimal (β, σ, C)
    print(f"\n3. Grid Search over (β, σ, C)...")
    print(f"   β values: {beta_values}")
    print(f"   C values: {C_values}")
    print(f"   CV folds: {cv_folds}")
    
    grid_results = grid_search_kernel_params(
        X_train=X_train,
        y_train=y_train,
        beta_values=beta_values,
        C_values=C_values,
        cv=cv_folds,
        use_adaptive_sigma=True,
        sigma_scale_factor=3.0
    )
    
    best_params = grid_results['best_params']
    best_cv_score = grid_results['best_score']
    
    # Evaluate all configurations on test set
    print(f"\n4. Evaluating all configurations on test set...")
    results = []
    
    for config in grid_results['all_results']:
        beta = config['beta']
        sigma = config['sigma']
        C = config['C']
        
        # Fit kernel with this configuration
        kernel = SpectralFlatteningKernel(beta=beta, sigma=sigma, shrinkage=True)
        kernel.fit(X_train)
        
        K_train = kernel.compute_kernel_matrix(X_train)
        K_test = kernel.compute_kernel_matrix(X_test, X_train)
        
        # Evaluate on test set
        test_metrics = evaluate_kernel_svm(
            K_train=K_train,
            K_test=K_test,
            y_train=y_train,
            y_test=y_test,
            C=C
        )
        
        # Compute diagnostics
        alignment = compute_kernel_target_alignment(K_train, y_train)
        eff_rank = compute_effective_rank(K_train)
        cond_num = kernel.get_condition_number()
        
        result = {
            'beta': float(beta),
            'sigma': float(sigma),
            'C': float(C),
            'cv_score': float(config['cv_score']),
            'cv_fold_scores': [float(x) for x in config['cv_fold_scores']],
            'cv_std': float(config['cv_std']),
            'test_accuracy': float(test_metrics['test_accuracy']),
            'train_accuracy': float(test_metrics['train_accuracy']),
            'balanced_accuracy': float(test_metrics['balanced_accuracy']),
            'f1_macro': float(test_metrics['f1_macro']),
            'f1_weighted': float(test_metrics['f1_weighted']),
            'kernel_alignment': float(alignment),
            'effective_rank': float(eff_rank),
            'condition_number': float(cond_num)
        }
        
        results.append(result)
        
        print(f"   β={beta:.3f}, σ={sigma:.2f}, C={C:.1f}: "
              f"CV={config['cv_score']:.4f}, Test={test_metrics['test_accuracy']:.4f}, "
              f"F1={test_metrics['f1_macro']:.4f}")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save CSV
    csv_path = f'{output_dir}/results.csv'
    df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"\n5. Saved results to {csv_path}")
    
    # Save JSON with full details (including CV fold scores)
    json_data = {
        'metadata': {
            'dataset': dataset_name,
            'dataset_display': dataset_display,
            'model': model_name,
            'timestamp': datetime.now().isoformat(),
            'cv_folds': cv_folds
        },
        'dataset_stats': {
            'n_train': int(X_train.shape[0]),
            'n_test': int(X_test.shape[0]),
            'n_features': int(X_train.shape[1]),
            'n_classes': int(len(np.unique(y_train)))
        },
        'spectral_properties': {
            'condition_number': float(condition_number),
            'effective_rank': float(effective_rank_orig),
            'eigenvalues': eigenvalues.tolist(),
            'cumulative_variance_top_10': float(cumvar[9]),
            'cumulative_variance_top_50': float(cumvar[49]),
            'cumulative_variance_top_100': float(cumvar[99])
        },
        'grid_search_config': {
            'beta_values': beta_values,
            'C_values': C_values,
            'cv_folds': cv_folds,
            'adaptive_sigma': True,
            'sigma_scale_factor': 3.0
        },
        'results': results  # Already includes cv_fold_scores and cv_std
    }
    
    json_path = f'{output_dir}/results.json'
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"   Saved JSON with CV details to {json_path}")
    
    # Generate summary
    print(f"\n6. Generating summary...")
    
    baseline = df[df['beta'] == 0.0].iloc[0]
    # CRITICAL: Select best by CV score, not test accuracy (avoid selecting overfitted models)
    best = df.loc[df['cv_score'].idxmax()]
    
    # Also track which β had best test accuracy for comparison
    best_test = df.loc[df['test_accuracy'].idxmax()]
    
    summary_lines = [
        f"EVALUATION SUMMARY: {model_name.upper()}",
        "=" * 80,
        f"Dataset: {dataset_display}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "DATASET STATISTICS",
        "-" * 80,
        f"Training samples:   {X_train.shape[0]}",
        f"Test samples:       {X_test.shape[0]}",
        f"Feature dimension:  {X_train.shape[1]}",
        f"Number of classes:  {len(np.unique(y_train))}",
        "",
        "SPECTRAL PROPERTIES (ORIGINAL FEATURES)",
        "-" * 80,
        f"Condition number:   {condition_number:.2e}",
        f"Effective rank:     {effective_rank_orig:.1f} / {X_train.shape[1]}",
        f"Top 10 PCs variance: {cumvar[9]:.2%}",
        f"Top 50 PCs variance: {cumvar[49]:.2%}",
        f"Top 100 PCs variance: {cumvar[99]:.2%}",
        "",
        "GRID SEARCH CONFIGURATION",
        "-" * 80,
        f"β values tested:    {beta_values}",
        f"C values tested:    {C_values}",
        f"CV folds:           {cv_folds}",
        f"Adaptive σ:         Yes (median_distance / 3.0)",
        f"Total configs:      {len(results)}",
        "",
        "BASELINE (β=0.0, RBF KERNEL)",
        "-" * 80,
        f"σ (adaptive):       {baseline['sigma']:.2f}",
        f"C (optimal):        {baseline['C']:.2f}",
        f"CV Score:           {baseline['cv_score']:.4f}",
        f"Test Accuracy:      {baseline['test_accuracy']:.4f}",
        f"Train Accuracy:     {baseline['train_accuracy']:.4f}",
        f"Balanced Accuracy:  {baseline['balanced_accuracy']:.4f}",
        f"F1 Macro:           {baseline['f1_macro']:.4f}",
        f"Kernel Alignment:   {baseline['kernel_alignment']:.4f}",
        f"Effective Rank:     {baseline['effective_rank']:.1f}",
        f"Condition Number:   {baseline['condition_number']:.2e}",
        "",
        "BEST CONFIGURATION (by CV score)",
        "-" * 80,
        f"β (optimal):        {best['beta']:.3f}",
        f"σ (adaptive):       {best['sigma']:.2f}",
        f"C (optimal):        {best['C']:.2f}",
        f"CV Score:           {best['cv_score']:.4f}",
        f"Test Accuracy:      {best['test_accuracy']:.4f}",
        f"Train Accuracy:     {best['train_accuracy']:.4f}",
        f"Balanced Accuracy:  {best['balanced_accuracy']:.4f}",
        f"F1 Macro:           {best['f1_macro']:.4f}",
        f"Kernel Alignment:   {best['kernel_alignment']:.4f}",
        f"Effective Rank:     {best['effective_rank']:.1f}",
        f"Condition Number:   {best['condition_number']:.2e}",
        "",
        "BEST TEST ACCURACY (may be overfitted)",
        "-" * 80,
        f"β:                  {best_test['beta']:.3f}",
        f"Test Accuracy:      {best_test['test_accuracy']:.4f}",
        f"Train Accuracy:     {best_test['train_accuracy']:.4f}",
        f"CV Score:           {best_test['cv_score']:.4f}",
        f"⚠ Train-Test gap:   {(best_test['train_accuracy'] - best_test['test_accuracy'])*100:.1f}pp (overfitting indicator)",
        "",
        "IMPROVEMENT OVER BASELINE",
        "-" * 80,
        f"Accuracy:           {(best['test_accuracy'] - baseline['test_accuracy'])*100:+.2f} pp",
        f"F1 Macro:           {(best['f1_macro'] - baseline['f1_macro'])*100:+.2f} pp",
        f"Alignment:          {(best['kernel_alignment'] - baseline['kernel_alignment']):.4f}",
        f"Cond. Num. Reduction: {baseline['condition_number'] / best['condition_number']:.2f}×",
        "",
        "PERFORMANCE ACROSS β VALUES",
        "-" * 80,
    ]
    
    # Add per-β performance table
    summary_lines.append(f"{'β':<8} {'σ':<10} {'C':<8} {'CV Score':<10} {'Test Acc':<10} {'Train Acc':<11} {'Gap':<10}")
    summary_lines.append("-" * 80)
    
    for beta in sorted(df['beta'].unique()):
        beta_best = df[df['beta'] == beta].loc[df[df['beta'] == beta]['cv_score'].idxmax()]
        gap = beta_best['train_accuracy'] - beta_best['test_accuracy']
        summary_lines.append(
            f"{beta_best['beta']:<8.3f} "
            f"{beta_best['sigma']:<10.2f} "
            f"{beta_best['C']:<8.2f} "
            f"{beta_best['cv_score']:<10.4f} "
            f"{beta_best['test_accuracy']:<10.4f} "
            f"{beta_best['train_accuracy']:<11.4f} "
            f"{gap*100:<10.1f}"
        )
    
    summary_lines.append("")
    summary_lines.append("=" * 80)
    
    summary_text = "\n".join(summary_lines)
    
    # Save summary
    summary_path = f'{output_dir}/summary.txt'
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    
    print(summary_text)
    print(f"\n7. Saved summary to {summary_path}")
    
    # Generate plots WITH ERROR BARS
    print(f"\n8. Generating plots with error bars from CV folds...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Aggregate data per β (mean and std across all C values)
    beta_stats = []
    for beta in sorted(df['beta'].unique()):
        beta_df = df[df['beta'] == beta]
        # Use best C for this beta
        best_for_beta = beta_df.loc[beta_df['cv_score'].idxmax()]
        beta_stats.append({
            'beta': beta,
            'test_accuracy_mean': best_for_beta['test_accuracy'],
            'cv_score_mean': best_for_beta['cv_score'],
            'cv_score_std': best_for_beta['cv_std'],
            'alignment_mean': best_for_beta['kernel_alignment'],
            'condition_number': best_for_beta['condition_number'],
            'effective_rank': best_for_beta['effective_rank']
        })
    beta_stats = pd.DataFrame(beta_stats)
    
    # Plot 1: Test Accuracy vs β WITH ERROR BARS
    ax = axes[0, 0]
    ax.errorbar(beta_stats['beta'], beta_stats['cv_score_mean'], 
                yerr=beta_stats['cv_score_std'],
                marker='o', markersize=8, linewidth=2, capsize=5, capthick=2,
                label='CV Score (mean ± std)', color='blue')
    ax.plot(beta_stats['beta'], beta_stats['test_accuracy_mean'],
            marker='s', markersize=8, linewidth=2, linestyle='--',
            label='Test Accuracy', color='green', alpha=0.7)
    ax.axhline(baseline['test_accuracy'], color='red', linestyle=':', 
               label='Baseline Test', alpha=0.7, linewidth=2)
    ax.set_xlabel('β (Spectral Flattening)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax.set_title('Accuracy vs β (with CV error bars)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    # Plot 2: Kernel-Target Alignment vs β
    ax = axes[0, 1]
    ax.plot(beta_stats['beta'], beta_stats['alignment_mean'],
            marker='D', markersize=8, linewidth=2, color='purple')
    ax.axhline(baseline['kernel_alignment'], color='red', linestyle=':', 
               label='Baseline', alpha=0.7, linewidth=2)
    ax.set_xlabel('β (Spectral Flattening)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Kernel-Target Alignment', fontsize=11, fontweight='bold')
    ax.set_title('Kernel Alignment vs β', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    # Plot 3: Condition Number vs β (log scale)
    ax = axes[1, 0]
    ax.semilogy(beta_stats['beta'], beta_stats['condition_number'],
                marker='^', markersize=8, linewidth=2, color='orange')
    ax.axhline(baseline['condition_number'], color='red', linestyle=':', 
               label='Baseline', alpha=0.7, linewidth=2)
    ax.set_xlabel('β (Spectral Flattening)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Condition Number (log scale)', fontsize=11, fontweight='bold')
    ax.set_title('Condition Number vs β', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=9)
    
    # Plot 4: Effective Rank vs β
    ax = axes[1, 1]
    ax.plot(beta_stats['beta'], beta_stats['effective_rank'],
            marker='*', markersize=10, linewidth=2, color='brown')
    ax.axhline(baseline['effective_rank'], color='red', linestyle=':', 
               label='Baseline', alpha=0.7, linewidth=2)
    ax.set_xlabel('β (Spectral Flattening)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Effective Rank', fontsize=11, fontweight='bold')
    ax.set_title('Effective Rank vs β', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    plot_path = f'{output_dir}/plots.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved plots with error bars to {plot_path}")
    
    # Return results for further analysis
    return {
        'dataset': dataset_name,
        'model_name': model_name,
        'dataframe': df,
        'best_params': best_params,
        'best_cv_score': best_cv_score,
        'baseline_test_acc': baseline['test_accuracy'],
        'best_test_acc': best['test_accuracy'],
        'improvement': (best['test_accuracy'] - baseline['test_accuracy']) * 100
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate spectral flattening on real data')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['cub', 'stanford_cars'],
                       help='Dataset to evaluate on')
    parser.add_argument('--model', type=str, 
                       choices=['resnet50', 'vit_b_16', 'convnext_base'],
                       help='Model architecture to evaluate')
    parser.add_argument('--all', action='store_true',
                       help='Evaluate all models sequentially')
    parser.add_argument('--cv', type=int, default=3,
                       help='Number of CV folds (default: 3)')
    
    args = parser.parse_args()
    
    dataset_display = {'cub': 'CUB-200-2011', 'stanford_cars': 'Stanford Cars'}[args.dataset]
    
    if args.all:
        models = ['resnet50', 'vit_b_16', 'convnext_base']
        print("="*80)
        print(f"EVALUATING ALL MODELS ON {dataset_display.upper()}")
        print("="*80)
        
        all_results = {}
        for model in models:
            # Check if features exist
            train_path = f'results/features/{args.dataset}/{model}_train.npz'
            test_path = f'results/features/{args.dataset}/{model}_test.npz'
            
            if not os.path.exists(train_path) or not os.path.exists(test_path):
                print(f"\nSkipping {model}: Features not found")
                print(f"   Run: python scripts/extract_all_features.py --dataset {args.dataset} --models {model}")
                continue
            
            result = evaluate_model(args.dataset, model, cv_folds=args.cv)
            all_results[model] = result
            print(f"\nCompleted {model}")
        
        # Summary across models
        print("\n" + "="*80)
        print(f"SUMMARY: {dataset_display.upper()}")
        print("="*80)
        for model, res in all_results.items():
            print(f"\n{model}:")
            print(f"  Best β:         {res['best_params']['beta']:.3f}")
            print(f"  Test Accuracy:  {res['best_test_acc']:.4f}")
            print(f"  Improvement:    {res['improvement']:+.2f} pp")
        
    elif args.model:
        # Check if features exist
        train_path = f'results/features/{args.dataset}/{args.model}_train.npz'
        test_path = f'results/features/{args.dataset}/{args.model}_test.npz'
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print(f"Error: Features not found for {args.model} on {dataset_display}")
            print(f"   Expected:")
            print(f"     - {train_path}")
            print(f"     - {test_path}")
            print(f"\n   Run: python scripts/extract_all_features.py --dataset {args.dataset} --models {args.model}")
            sys.exit(1)
        
        evaluate_model(args.dataset, args.model, cv_folds=args.cv)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
