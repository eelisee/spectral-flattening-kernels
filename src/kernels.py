"""
Spectral flattening kernel implementations
Based on idea4.txt theoretical framework
"""

import numpy as np
from scipy.linalg import eigh
from sklearn.covariance import LedoitWolf
from typing import Tuple, Optional


class SpectralFlatteningKernel:
    """
    Implements spectral flattening kernels k_β with parameter β ∈ [0, 0.5]
    
    k_β(x, x') = exp(-||T_β(x - x')||² / (2σ²))
    where T_β = U Λ^(-β) U^T
    
    β=0: isotropic RBF
    β=0.5: whitening
    """
    
    def __init__(self, beta: float = 0.25, sigma: float = 1.0, 
                 shrinkage: bool = True):
        """
        Args:
            beta: Spectral flattening parameter in [0, 0.5]
            sigma: Bandwidth parameter
            shrinkage: Use Ledoit-Wolf shrinkage for covariance estimation
        """
        if beta < 0 or beta > 0.5:
            raise ValueError("beta must be in [0, 0.5]")
        
        self.beta = beta
        self.sigma = sigma
        self.shrinkage = shrinkage
        self.U = None
        self.Lambda = None
        self.T_beta = None
        self.Sigma = None
    
    def fit(self, X: np.ndarray):
        """
        Fit the kernel by estimating covariance and computing T_β transform
        
        Args:
            X: Training features (n_samples, n_features)
        """
        # Estimate covariance with optional shrinkage
        if self.shrinkage:
            lw = LedoitWolf()
            self.Sigma = lw.fit(X).covariance_
        else:
            self.Sigma = np.cov(X.T)
        
        # Eigen-decomposition: Σ = U Λ U^T
        eigenvalues, eigenvectors = eigh(self.Sigma)
        
        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Ensure positive eigenvalues (numerical stability)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        
        self.Lambda = eigenvalues
        self.U = eigenvectors
        
        # Compute T_β = U Λ^(-β) U^T
        Lambda_beta = np.diag(eigenvalues ** (-self.beta))
        self.T_beta = self.U @ Lambda_beta @ self.U.T
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply T_β transformation to features
        
        Args:
            X: Features to transform (n_samples, n_features)
            
        Returns:
            Transformed features
        """
        if self.T_beta is None:
            raise ValueError("Must call fit() before transform()")
        
        return X @ self.T_beta.T
    
    def compute_kernel_matrix(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute kernel matrix K[i,j] = k_β(X[i], Y[j])
        
        Args:
            X: First set of samples (n, d)
            Y: Second set of samples (m, d). If None, Y = X
            
        Returns:
            Kernel matrix (n, m)
        """
        if Y is None:
            Y = X
        
        # Transform both sets
        X_trans = self.transform(X)
        Y_trans = self.transform(Y)
        
        # Compute pairwise squared distances
        # ||x - y||² = ||x||² + ||y||² - 2x·y
        X_norm_sq = np.sum(X_trans ** 2, axis=1, keepdims=True)
        Y_norm_sq = np.sum(Y_trans ** 2, axis=1, keepdims=True)
        
        dist_sq = X_norm_sq + Y_norm_sq.T - 2 * X_trans @ Y_trans.T
        
        # Gaussian kernel
        K = np.exp(-dist_sq / (2 * self.sigma ** 2))
        
        return K
    
    def get_transformed_covariance(self) -> np.ndarray:
        """
        Get the transformed covariance Σ_β = U Λ^(1-2β) U^T
        
        Returns:
            Transformed covariance matrix
        """
        if self.U is None:
            raise ValueError("Must call fit() first")
        
        Lambda_transformed = np.diag(self.Lambda ** (1 - 2 * self.beta))
        Sigma_beta = self.U @ Lambda_transformed @ self.U.T
        
        return Sigma_beta
    
    def get_condition_number(self) -> float:
        """
        Get condition number κ(Σ_β) = κ(Σ)^(1-2β)
        
        Returns:
            Condition number
        """
        if self.Lambda is None:
            raise ValueError("Must call fit() first")
        
        kappa_original = self.Lambda[0] / self.Lambda[-1]
        kappa_transformed = kappa_original ** (1 - 2 * self.beta)
        
        return kappa_transformed
    
    def get_spectral_info(self) -> dict:
        """Get spectral information about the kernel"""
        if self.Lambda is None:
            raise ValueError("Must call fit() first")
        
        return {
            'eigenvalues': self.Lambda.copy(),
            'transformed_eigenvalues': self.Lambda ** (1 - 2 * self.beta),
            'condition_number_original': self.Lambda[0] / self.Lambda[-1],
            'condition_number_transformed': self.get_condition_number(),
            'effective_rank': np.sum(self.Lambda) ** 2 / np.sum(self.Lambda ** 2)
        }


def compute_kernel_target_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """
    Compute kernel-target alignment (Cristianini et al.)
    
    A(K, Y) = <K, YY^T>_F / (||K||_F ||YY^T||_F)
    
    Args:
        K: Kernel matrix (n, n)
        y: Labels (n,)
        
    Returns:
        Alignment score in [0, 1]
    """
    n = len(y)
    
    # Create centered label matrix
    # For multi-class: Y is one-hot, YY^T is the ideal kernel
    classes = np.unique(y)
    Y = np.zeros((n, len(classes)))
    for i, c in enumerate(classes):
        Y[y == c, i] = 1
    
    YYT = Y @ Y.T
    
    # Frobenius inner product and norms
    alignment = np.sum(K * YYT) / (np.linalg.norm(K, 'fro') * np.linalg.norm(YYT, 'fro'))
    
    return alignment


def compute_effective_rank(K: np.ndarray) -> float:
    """
    Compute effective rank (stable rank) of kernel matrix
    
    Effective rank = trace(K)² / ||K||_F²
    
    Args:
        K: Kernel matrix
        
    Returns:
        Effective rank
    """
    trace_K = np.trace(K)
    frob_norm_sq = np.sum(K ** 2)
    
    return trace_K ** 2 / frob_norm_sq


def compute_rkhs_separation(X_class1: np.ndarray, X_class2: np.ndarray, 
                           kernel: SpectralFlatteningKernel) -> float:
    """
    Compute approximate RKHS separation between class means (small-signal regime)
    
    ||μ_k^(+) - μ_k^(-)||²_H ≈ Δ^T M_β Δ / σ²
    
    Args:
        X_class1: Samples from class 1 (n1, d)
        X_class2: Samples from class 2 (n2, d)
        kernel: Fitted spectral flattening kernel
        
    Returns:
        Approximate RKHS separation
    """
    # Class means
    mu1 = np.mean(X_class1, axis=0)
    mu2 = np.mean(X_class2, axis=0)
    delta = mu1 - mu2
    
    # M_β = U Λ^(-2β) U^T
    M_beta = kernel.U @ np.diag(kernel.Lambda ** (-2 * kernel.beta)) @ kernel.U.T
    
    # RKHS separation (leading order)
    separation = (delta @ M_beta @ delta) / (kernel.sigma ** 2)
    
    return separation
