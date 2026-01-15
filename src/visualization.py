"""
Advanced visualization utilities for kernel analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
from scipy.stats import gaussian_kde
from typing import List, Dict, Optional, Tuple
import matplotlib.patches as mpatches


def plot_kernel_heatmap(K: np.ndarray, y: np.ndarray, title: str = "Kernel Matrix",
                       save_path: Optional[str] = None):
    """
    Plot kernel matrix heatmap with class ordering
    
    Args:
        K: Kernel matrix
        y: Labels for ordering
        title: Plot title
        save_path: Path to save figure
    """
    # Sort by labels for better visualization
    idx = np.argsort(y)
    K_sorted = K[np.ix_(idx, idx)]
    y_sorted = y[idx]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(K_sorted, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax, label='Kernel Value')
    
    # Add class boundaries
    class_changes = np.where(np.diff(y_sorted) != 0)[0] + 1
    for pos in class_changes:
        ax.axhline(pos, color='red', linewidth=0.5, alpha=0.5)
        ax.axvline(pos, color='red', linewidth=0.5, alpha=0.5)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Sample Index (sorted by class)')
    ax.set_ylabel('Sample Index (sorted by class)')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix_comparison(y_true: np.ndarray, 
                                    predictions: Dict[str, np.ndarray],
                                    class_names: Optional[List[str]] = None,
                                    save_path: Optional[str] = None):
    """
    Plot confusion matrices for multiple methods side by side
    
    Args:
        y_true: True labels
        predictions: Dictionary of method_name -> predictions
        class_names: Optional class names
        save_path: Path to save figure
    """
    from sklearn.metrics import confusion_matrix
    
    n_methods = len(predictions)
    fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 5))
    
    if n_methods == 1:
        axes = [axes]
    
    for ax, (method_name, y_pred) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize by row (true class)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        im = ax.imshow(cm_norm, cmap='Blues', aspect='auto')
        ax.set_title(f'{method_name}\n', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('True Class')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_decision_boundary_2d(X: np.ndarray, y: np.ndarray, 
                              kernel_transform_func,
                              beta: float,
                              title: str = "Decision Boundary",
                              save_path: Optional[str] = None):
    """
    Plot decision boundary for 2D data
    
    Args:
        X: 2D features
        y: Labels
        kernel_transform_func: Function to compute kernel
        beta: Flattening parameter
        title: Plot title
        save_path: Path to save figure
    """
    from sklearn.svm import SVC
    
    # Create mesh
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Train SVM
    K = kernel_transform_func(X, beta)
    svm = SVC(kernel='precomputed', C=1.0)
    svm.fit(K, y)
    
    # Predict on mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    K_mesh = kernel_transform_func(mesh_points, beta, X)
    Z = svm.predict(K_mesh)
    Z = Z.reshape(xx.shape)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', 
                        edgecolors='black', s=50)
    ax.set_title(f'{title} (β={beta:.2f})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    plt.colorbar(scatter, ax=ax)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_eigenvalue_comparison(eigenvalues_dict: Dict[str, np.ndarray],
                               save_path: Optional[str] = None):
    """
    Compare eigenvalue spectra for different transformations
    
    Args:
        eigenvalues_dict: Dictionary of transformation_name -> eigenvalues
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Linear scale (top components)
    ax = axes[0, 0]
    for name, eigvals in eigenvalues_dict.items():
        ax.plot(eigvals[:100], label=name, linewidth=2, alpha=0.7)
    ax.set_xlabel('Eigenvalue Index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('Eigenvalue Spectrum (Linear, Top 100)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Log scale (all)
    ax = axes[0, 1]
    for name, eigvals in eigenvalues_dict.items():
        ax.semilogy(eigvals, label=name, linewidth=2, alpha=0.7)
    ax.set_xlabel('Eigenvalue Index')
    ax.set_ylabel('Eigenvalue (log scale)')
    ax.set_title('Eigenvalue Spectrum (Log Scale)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Cumulative variance explained
    ax = axes[1, 0]
    for name, eigvals in eigenvalues_dict.items():
        cumvar = np.cumsum(eigvals) / np.sum(eigvals)
        ax.plot(cumvar[:200], label=name, linewidth=2, alpha=0.7)
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90%')
    ax.axhline(y=0.95, color='gray', linestyle=':', alpha=0.5, label='95%')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Variance Explained')
    ax.set_title('Cumulative Variance', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Condition numbers
    ax = axes[1, 1]
    names = list(eigenvalues_dict.keys())
    cond_numbers = [eigvals[0] / eigvals[-1] for eigvals in eigenvalues_dict.values()]
    bars = ax.bar(names, cond_numbers, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Condition Number')
    ax.set_title('Condition Number Comparison', fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, cond_numbers):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1e}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_performance_vs_beta(results: List[Dict],
                             metrics: List[str] = ['test_accuracy', 'kernel_alignment'],
                             save_path: Optional[str] = None):
    """
    Plot multiple metrics vs beta parameter
    
    Args:
        results: List of result dictionaries with 'beta' key
        metrics: List of metric names to plot
        save_path: Path to save figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    betas = np.array([r['beta'] for r in results])
    
    for ax, metric in zip(axes, metrics):
        values = np.array([r[metric] for r in results])
        
        ax.plot(betas, values, 'o-', linewidth=2, markersize=10)
        ax.set_xlabel('β (Spectral Flattening)', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{metric.replace("_", " ").title()} vs. β', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Mark best value
        best_idx = np.argmax(values)
        ax.axvline(x=betas[best_idx], color='red', linestyle='--', 
                  alpha=0.5, label=f'Best β={betas[best_idx]:.3f}')
        ax.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_statistical_comparison(comparison_results: Dict[str, Dict],
                                save_path: Optional[str] = None):
    """
    Visualize statistical test results
    
    Args:
        comparison_results: Dictionary of comparison_name -> test_results
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    names = list(comparison_results.keys())
    p_values = [comparison_results[name]['p_value'] for name in names]
    
    colors = ['green' if p < 0.05 else 'orange' if p < 0.1 else 'red' 
             for p in p_values]
    
    bars = ax.barh(names, p_values, color=colors, alpha=0.7, edgecolor='black')
    ax.axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='α=0.05')
    ax.axvline(x=0.1, color='orange', linestyle='--', linewidth=2, label='α=0.10')
    ax.set_xlabel('p-value', fontsize=12)
    ax.set_title('Statistical Significance Tests', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add p-value labels
    for bar, p in zip(bars, p_values):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f' p={p:.4f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_learning_curves(train_sizes: np.ndarray,
                         train_scores_dict: Dict[str, np.ndarray],
                         test_scores_dict: Dict[str, np.ndarray],
                         save_path: Optional[str] = None):
    """
    Plot learning curves for multiple methods
    
    Args:
        train_sizes: Array of training set sizes
        train_scores_dict: Dictionary of method -> train scores
        test_scores_dict: Dictionary of method -> test scores
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method_name in train_scores_dict.keys():
        train_scores = train_scores_dict[method_name]
        test_scores = test_scores_dict[method_name]
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        ax.plot(train_sizes, train_mean, 'o--', label=f'{method_name} (train)', alpha=0.7)
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
        
        ax.plot(train_sizes, test_mean, 'o-', label=f'{method_name} (test)', linewidth=2)
        ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2)
    
    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Learning Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_decision_boundaries_2d(X_train: np.ndarray, y_train: np.ndarray, 
                                X_test: np.ndarray, y_test: np.ndarray,
                                beta_values: List[float], 
                                save_path: Optional[str] = None):
    """
    Plot SVM decision boundaries for 2D data with different β values
    Shows the classic colored contour plots demonstrating spectral flattening effect
    
    Args:
        X_train: 2D training features (n_train, 2)
        y_train: Training labels (n_train,)
        X_test: 2D test features (n_test, 2)
        y_test: Test labels (n_test,)
        beta_values: List of β values to compare
        save_path: Path to save figure
    """
    from scipy.spatial.distance import pdist
    from src.kernels import SpectralFlatteningKernel
    from src.svm_utils import KernelSVM
    
    n_betas = len(beta_values)
    fig, axes = plt.subplots(1, n_betas, figsize=(6*n_betas, 5))
    
    if n_betas == 1:
        axes = [axes]
    
    # Create mesh for contour plot
    x_min, x_max = X_train[:, 0].min() - 5, X_train[:, 0].max() + 5
    y_min, y_max = X_train[:, 1].min() - 2, X_train[:, 1].max() + 2
    h = 0.1  # Step size in mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    contourf_plot = None  # Store for colorbar
    
    for idx, beta in enumerate(beta_values):
        ax = axes[idx]
        
        # Fit kernel with adaptive sigma
        kernel = SpectralFlatteningKernel(beta=beta, sigma=1.0, shrinkage=False)
        kernel.fit(X_train)
        
        # Adaptive sigma in transformed space
        X_train_transformed = kernel.transform(X_train)
        distances = pdist(X_train_transformed, metric='euclidean')
        sigma_adaptive = np.median(distances) / 3
        
        kernel = SpectralFlatteningKernel(beta=beta, sigma=sigma_adaptive, shrinkage=False)
        kernel.fit(X_train)
        
        # Compute kernel matrices
        K_train = kernel.compute_kernel_matrix(X_train)
        
        # Train SVM
        svm_model = KernelSVM(C=1.0)
        svm_model.fit(K_train, y_train)
        
        # Predict on mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        K_mesh = kernel.compute_kernel_matrix(mesh_points, X_train)
        Z = svm_model.predict(K_mesh)
        Z = Z.reshape(xx.shape)
        
        # Get decision function values for smooth gradient
        decision_values = svm_model.svm.decision_function(K_mesh)
        
        # For binary classification, decision_function is 1D
        # For multi-class, take max over classes for visualization
        if len(decision_values.shape) > 1 and decision_values.shape[1] > 1:
            decision_values = np.max(decision_values, axis=1)
        
        decision_values = decision_values.reshape(xx.shape)
        
        # Plot smooth decision boundary with color gradient
        levels = np.linspace(decision_values.min(), decision_values.max(), 50)
        contourf_plot = ax.contourf(xx, yy, decision_values, levels=levels, 
                                     cmap='RdBu_r', alpha=0.7)
        
        # Plot decision boundary line (where decision = 0)
        ax.contour(xx, yy, decision_values, levels=[0], 
                  colors='black', linewidths=2.5, linestyles='solid')
        
        # Plot margins (decision = ±1) if binary
        if len(np.unique(y_train)) == 2:
            ax.contour(xx, yy, decision_values, levels=[-1, 1], 
                      colors='black', linewidths=1.5, linestyles='dashed', alpha=0.5)
        
        # Plot training points
        scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
                           cmap='RdBu_r', s=80, edgecolors='black', linewidth=1.5,
                           alpha=0.9, zorder=10)
        
        # Highlight support vectors (small points only)
        if hasattr(svm_model.svm, 'support_'):
            support_indices = svm_model.svm.support_
            ax.scatter(X_train[support_indices, 0], X_train[support_indices, 1],
                      s=30, c='gold', edgecolors='black', linewidth=0.8,
                      zorder=11, marker='o', label='Support Vectors')
        
        # Test accuracy
        K_test = kernel.compute_kernel_matrix(X_test, X_train)
        y_pred = svm_model.predict(K_test)
        test_acc = np.mean(y_pred == y_test)
        
        # Styling
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Feature 1 (high variance)', fontsize=11)
        ax.set_ylabel('Feature 2 (low variance)', fontsize=11)
        ax.set_title(f'β = {beta:.3f}\nTest Acc: {test_acc:.1%}', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.2, linestyle='--')
        
        if idx == 0:
            ax.legend(loc='upper left', fontsize=9)
    
    # Add colorbar BELOW the plots
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.025])  # [left, bottom, width, height]
    cbar = fig.colorbar(contourf_plot, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('SVM Decision Function', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Decision boundary plot saved to {save_path}")
    plt.close()


def plot_spectral_flattening_effect_2d(X_train: np.ndarray, y_train: np.ndarray,
                                       beta_values: List[float],
                                       save_path: Optional[str] = None):
    """
    Comprehensive visualization showing HOW spectral flattening transforms the space.
    Shows original space, transformed space, covariance ellipses, and kernel distance contours.
    
    This is the didactic "explainer" plot that makes spectral flattening intuitive.
    
    Args:
        X_train: 2D training features (n_train, 2)
        y_train: Training labels (n_train,)
        beta_values: List of β values to compare
        save_path: Path to save figure
    """
    from src.kernels import SpectralFlatteningKernel
    from matplotlib.patches import Ellipse
    
    n_betas = len(beta_values)
    
    # Create figure with 3 rows: Original, Transformed, Covariance Ellipses
    fig = plt.figure(figsize=(6*n_betas, 15))
    gs = fig.add_gridspec(3, n_betas, hspace=0.3, wspace=0.3)
    
    # Compute covariance once
    Sigma = np.cov(X_train.T)
    
    for col_idx, beta in enumerate(beta_values):
        
        # Fit kernel
        kernel = SpectralFlatteningKernel(beta=beta, sigma=1.0, shrinkage=False)
        kernel.fit(X_train)
        
        # Transform data
        X_transformed = kernel.transform(X_train)
        
        # Get transformed covariance
        Sigma_beta = kernel.get_transformed_covariance()
        
        # ========================================
        # ROW 1: Original Space
        # ========================================
        ax1 = fig.add_subplot(gs[0, col_idx])
        
        # Plot points
        for label in np.unique(y_train):
            mask = y_train == label
            ax1.scatter(X_train[mask, 0], X_train[mask, 1], 
                       s=50, alpha=0.6, edgecolors='black', linewidth=0.5,
                       label=f'Class {label}')
        
        ax1.set_xlabel('Feature 1 (high variance)', fontsize=11)
        ax1.set_ylabel('Feature 2 (low variance)', fontsize=11)
        ax1.set_title(f'Original Space\nβ = {beta:.3f}', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        if col_idx == 0:
            ax1.legend(fontsize=9)
        
        # ========================================
        # ROW 2: Transformed Space T_β(X)
        # ========================================
        ax2 = fig.add_subplot(gs[1, col_idx])
        
        # Plot transformed points
        for label in np.unique(y_train):
            mask = y_train == label
            ax2.scatter(X_transformed[mask, 0], X_transformed[mask, 1], 
                       s=50, alpha=0.6, edgecolors='black', linewidth=0.5,
                       label=f'Class {label}')
        
        # Compute aspect ratio
        x_range = X_transformed[:, 0].max() - X_transformed[:, 0].min()
        y_range = X_transformed[:, 1].max() - X_transformed[:, 1].min()
        aspect_ratio = y_range / x_range if x_range > 0 else 1.0
        
        ax2.set_xlabel('Transformed Feature 1', fontsize=11)
        ax2.set_ylabel('Transformed Feature 2', fontsize=11)
        ax2.set_title(f'Transformed Space $T_\\beta(X)$\n(Kernel Distance = Euclidean)', 
                     fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        # Add annotation
        cond_num = kernel.get_condition_number()
        ax2.text(0.95, 0.05, f'κ(Σ_β) = {cond_num:.1f}\nAspect: {aspect_ratio:.2f}',
                transform=ax2.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # ========================================
        # ROW 3: Covariance Ellipses + Kernel Distance Contours
        # ========================================
        ax3 = fig.add_subplot(gs[2, col_idx])
        
        # Choose a reference point (mean of first class)
        x0 = X_train[y_train == np.unique(y_train)[0]].mean(axis=0)
        
        # Create grid
        x_min, x_max = X_train[:, 0].min() - 5, X_train[:, 0].max() + 5
        y_min, y_max = X_train[:, 1].min() - 2, X_train[:, 1].max() + 2
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                            np.linspace(y_min, y_max, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        
        # Compute kernel distances from x0
        T_beta = kernel.T_beta
        diff = grid - x0
        dist_sq = np.sum((diff @ T_beta.T) ** 2, axis=1)
        dist_sq = dist_sq.reshape(xx.shape)
        
        # Plot kernel distance contours
        levels = np.percentile(dist_sq, [25, 50, 75, 90, 95])
        contours = ax3.contour(xx, yy, dist_sq, levels=levels, 
                              colors='purple', linewidths=1.5, alpha=0.6)
        ax3.clabel(contours, inline=True, fontsize=8, fmt='d²=%.1f')
        
        # Plot covariance ellipse for original covariance
        vals_orig, vecs_orig = np.linalg.eigh(Sigma)
        angle_orig = np.degrees(np.arctan2(vecs_orig[1, 1], vecs_orig[0, 1]))
        ellipse_orig = Ellipse(xy=(0, 0), 
                              width=4*np.sqrt(vals_orig[0]),
                              height=4*np.sqrt(vals_orig[1]),
                              angle=angle_orig,
                              facecolor='none', edgecolor='red', linewidth=2,
                              linestyle='--', alpha=0.7, label='Σ (original)')
        ax3.add_patch(ellipse_orig)
        
        # Plot covariance ellipse for transformed covariance
        vals_beta, vecs_beta = np.linalg.eigh(Sigma_beta)
        angle_beta = np.degrees(np.arctan2(vecs_beta[1, 1], vecs_beta[0, 1]))
        ellipse_beta = Ellipse(xy=(0, 0),
                              width=4*np.sqrt(vals_beta[0]),
                              height=4*np.sqrt(vals_beta[1]),
                              angle=angle_beta,
                              facecolor='none', edgecolor='blue', linewidth=2.5,
                              linestyle='-', alpha=0.9, label=f'Σ_β (β={beta:.2f})')
        ax3.add_patch(ellipse_beta)
        
        # Plot scatter (lighter)
        for label in np.unique(y_train):
            mask = y_train == label
            ax3.scatter(X_train[mask, 0], X_train[mask, 1], 
                       s=20, alpha=0.3, edgecolors='none')
        
        # Mark reference point
        ax3.scatter([x0[0]], [x0[1]], s=200, c='gold', marker='*', 
                   edgecolors='black', linewidth=1.5, zorder=10,
                   label='Reference point')
        
        ax3.set_xlabel('Feature 1', fontsize=11)
        ax3.set_ylabel('Feature 2', fontsize=11)
        ax3.set_title(f'Covariance Ellipses & Kernel Distance Contours\n(Purple: iso-distance lines)',
                     fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.axis('equal')
        ax3.set_xlim(x_min, x_max)
        ax3.set_ylim(y_min, y_max)
        ax3.legend(fontsize=9, loc='upper right')
    
    # Overall title
    fig.suptitle('Spectral Flattening Effect: Original → Transformed → Geometry', 
                fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Spectral flattening effect plot saved to {save_path}")
    plt.close()
