"""
SVM training and evaluation utilities
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from scipy.spatial.distance import pdist
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class KernelSVM:
    """Kernel SVM wrapper with precomputed kernels"""
    
    def __init__(self, C: float = 1.0):
        """
        Args:
            C: SVM regularization parameter
        """
        self.C = C
        self.svm = None
        self.support_vectors_ = None
    
    def fit(self, K_train: np.ndarray, y_train: np.ndarray):
        """
        Fit SVM with precomputed kernel
        
        Args:
            K_train: Precomputed kernel matrix (n_train, n_train)
            y_train: Training labels (n_train,)
        """
        self.svm = SVC(kernel='precomputed', C=self.C)
        self.svm.fit(K_train, y_train)
        return self
    
    def predict(self, K_test: np.ndarray) -> np.ndarray:
        """
        Predict on test data
        
        Args:
            K_test: Kernel between test and training samples (n_test, n_train)
            
        Returns:
            Predicted labels (n_test,)
        """
        return self.svm.predict(K_test)
    
    def score(self, K_test: np.ndarray, y_test: np.ndarray) -> float:
        """Compute accuracy"""
        y_pred = self.predict(K_test)
        return accuracy_score(y_test, y_pred)


def cross_validate_svm(K: np.ndarray, y: np.ndarray, C_values: List[float], 
                       cv: int = 5) -> Tuple[float, float, List[float]]:
    """
    Cross-validate to find best C parameter
    
    Args:
        K: Precomputed kernel matrix
        y: Labels
        C_values: List of C values to try
        cv: Number of CV folds
        
    Returns:
        best_C: Best C value
        best_score: Best CV score (mean)
        best_fold_scores: CV scores for best C (all folds)
    """
    best_C = C_values[0]
    best_score = 0
    best_fold_scores = []
    
    for C in C_values:
        svm = SVC(kernel='precomputed', C=C)
        scores = cross_val_score(svm, K, y, cv=cv, scoring='accuracy')
        mean_score = np.mean(scores)
        
        if mean_score > best_score:
            best_score = mean_score
            best_C = C
            best_fold_scores = scores.tolist()
    
    return best_C, best_score, best_fold_scores


def evaluate_kernel_svm(K_train: np.ndarray, K_test: np.ndarray, 
                       y_train: np.ndarray, y_test: np.ndarray,
                       C: float = 1.0) -> Dict[str, float]:
    """
    Train and evaluate kernel SVM
    
    Args:
        K_train: Training kernel matrix (n_train, n_train)
        K_test: Test kernel matrix (n_test, n_train)
        y_train: Training labels
        y_test: Test labels
        C: SVM regularization parameter
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Train SVM
    svm = KernelSVM(C=C)
    svm.fit(K_train, y_train)
    
    # Predict
    y_pred = svm.predict(K_test)
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    return {
        'test_accuracy': accuracy,
        'train_accuracy': svm.score(K_train, y_train),
        'balanced_accuracy': balanced_acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }


def grid_search_kernel_params(X_train: np.ndarray, y_train: np.ndarray,
                              beta_values: List[float], 
                              C_values: List[float], 
                              cv: int = 3,
                              use_adaptive_sigma: bool = True,
                              sigma_scale_factor: float = 3.0,
                              manual_sigma_values: Optional[List[float]] = None) -> Dict:
    """
    Grid search over kernel parameters (β) and SVM parameter C
    Uses adaptive bandwidth per β (recommended) or manual σ grid
    
    Args:
        X_train: Training features (n_train, d_features)
        y_train: Training labels (n_train,)
        beta_values: List of β values to try
        C_values: List of C values to try
        cv: Number of CV folds
        use_adaptive_sigma: If True, compute adaptive σ per β (recommended)
        sigma_scale_factor: Scaling factor for adaptive σ (median_distance / scale_factor)
        manual_sigma_values: If use_adaptive_sigma=False, list of σ values to try
        
    Returns:
        Dictionary with best parameters and scores
    """
    from src.kernels import SpectralFlatteningKernel
    
    best_score = 0
    best_params = {}
    all_results = []
    
    print(f"\n=== Grid Search: {len(beta_values)} β values × {len(C_values)} C values ===")
    if use_adaptive_sigma:
        print(f"Using adaptive σ per β (median_distance / {sigma_scale_factor})")
    else:
        if manual_sigma_values is None:
            raise ValueError("Must provide manual_sigma_values when use_adaptive_sigma=False")
        print(f"Using manual σ grid: {len(manual_sigma_values)} values")
    
    for beta_idx, beta in enumerate(beta_values):
        print(f"\nTesting β = {beta:.3f} ({beta_idx+1}/{len(beta_values)})...")
        
        if use_adaptive_sigma:
            # Compute adaptive σ for this β (like in synthetic_data_experiment.py)
            kernel_temp = SpectralFlatteningKernel(beta=beta, sigma=1.0, shrinkage=True)
            kernel_temp.fit(X_train)
            X_transformed = kernel_temp.transform(X_train)
            distances = pdist(X_transformed, metric='euclidean')
            sigma_adaptive = np.median(distances) / sigma_scale_factor
            
            sigma_values_to_test = [sigma_adaptive]
            print(f"  Adaptive σ = {sigma_adaptive:.2f}")
        else:
            # Use manual σ grid
            sigma_values_to_test = manual_sigma_values
        
        for sigma in sigma_values_to_test:
            # Fit kernel with final β and σ
            kernel = SpectralFlatteningKernel(beta=beta, sigma=sigma, shrinkage=True)
            kernel.fit(X_train)
            K = kernel.compute_kernel_matrix(X_train)
            
            # Find best C via cross-validation
            best_C, score, fold_scores = cross_validate_svm(K, y_train, C_values, cv=cv)
            
            result = {
                'beta': beta,
                'sigma': sigma,
                'C': best_C,
                'cv_score': score,
                'cv_fold_scores': fold_scores,  # NEW: Store individual fold scores
                'cv_std': float(np.std(fold_scores))  # NEW: Standard deviation for error bars
            }
            all_results.append(result)
            
            print(f"  σ={sigma:.2f}, best_C={best_C:.2f}, CV_score={score:.4f}")
            
            if score > best_score:
                best_score = score
                best_params = result.copy()
    
    print(f"\n=== Best Configuration ===")
    print(f"β = {best_params['beta']:.3f}, σ = {best_params['sigma']:.2f}, C = {best_params['C']:.2f}")
    print(f"CV Score = {best_score:.4f}")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': all_results
    }