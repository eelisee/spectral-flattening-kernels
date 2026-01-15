# Code Documentation & Walkthrough

**Project**: Spectral Flattening for Kernel Methods on Pretrained Features  
**Authors**: Elise Wolf  
**Date**: January 2026  
**Course**: ENSAE Advanced Machine Learning

---

## Code Originality Statement

This codebase represents a focused implementation of spectral flattening for kernel methods, combining theoretical concepts from literature with practical experimentation. The core contribution is the complete experimental pipeline and analysis.

### Summary of Code Origins

| Component | Implementation Status | External Dependencies | Notes |
|-----------|----------------------|----------------------|-------|
| **Spectral Flattening Kernel** | Original implementation of mathematical concept | NumPy/SciPy | Formula from literature, coding is original |
| **SVM Training & CV** | Standard wrapper pattern | scikit-learn SVC | Common CV/grid search approach |
| **Feature Extraction** | Standard PyTorch pipeline | torchvision.models | Following official examples/docs |
| **Data Loaders** | Adapted from dataset docs | PIL, pandas | Based on official CUB/Cars format specs |
| **Visualization** | Standard matplotlib patterns | matplotlib, seaborn | Common plotting approaches |
| **Synthetic Data** | Original experimental design | NumPy | Custom covariance structures |
| **Experiment Scripts** | Original experimental pipeline | - | Complete evaluation framework |

**Attribution Summary**:
- No code copied from GitHub repositories
- Standard library usage follows official documentation (NumPy, PyTorch, scikit-learn)
- Dataset loaders based on official format specifications
- Pretrained models used only for feature extraction (torchvision ImageNet-1K weights)
- Mathematical formulas from kernel methods literature (Hofmann et al., 2008; Schölkopf & Smola, 2002)

---

## Code Structure & Walkthrough

### Directory Overview

```
advancedml/
├── src/                      # Core implementation
│   ├── kernels.py            # Spectral flattening kernel
│   ├── svm_utils.py          # SVM training & CV
│   ├── feature_extractor.py  # Deep feature extraction
│   ├── data_loader.py        # Dataset loading
│   ├── visualization.py      # Publication plots
│   └── statistical_tests.py  # Significance testing
│
├── scripts/                  # Experiment pipeline
│   ├── synthetic_data_test.py         # Phase 1: Theory validation
│   ├── synthetic_data_experiment.py   # Phase 1: Synthetic plots
│   ├── grid_search_test.py            # Phase 1: Grid Search
│   ├── covariance_structure_test.py   # Phase 1: Robustness tests
│   ├── download_datasets.py           # Phase 2: Data download
│   ├── extract_all_features.py        # Phase 2: Feature extraction
│   └── evaluate_real_data.py          # Phase 3: Real data evaluation
│
└── results/                  # Generated outputs
    ├── synthetic_data_test/
    ├── synthetic_data_experiment/
    ├── covariance_structure_test/
    ├── features/             # Cached features
    └── real_data/            # Per-model results
```

---

## Phase 1: Synthetic Validation

### `scripts/synthetic_data_test.py`

**Purpose**: Validate theoretical predictions on controlled synthetic data

**Key steps**:
1. Generate synthetic Gaussian data with known covariance structure
2. Inject class-discriminative signal in specific eigenspace directions
3. Test 4 scenarios:
   - LOW-variance signal (should improve with flattening)
   - HIGH-variance signal (should NOT improve)
   - Anisotropy sensitivity (improvement scales with condition number)
   - Sample size robustness (consistent across 25-200 samples/class)
4. Compare baseline (β=0) vs optimal (β>0) via cross-validation
5. Output: `results/synthetic_data_test/validation_results.csv` with PASS/FAIL

**Core implementation** (`src/kernels.py`):
```python
class SpectralFlatteningKernel:
    def compute_kernel(self, X, beta=0.0, sigma=None):
        # 1. Estimate covariance: Σ = (1/n) X^T X
        Sigma = np.cov(X.T) + self.epsilon * np.eye(d)
        
        # 2. Eigendecomposition: Σ = U Λ U^T
        eigenvalues, U = np.linalg.eigh(Sigma)
        
        # 3. Spectral transformation: Λ^(-β/2)
        Lambda_beta = np.diag((eigenvalues + self.epsilon) ** (-beta / 2))
        
        # 4. Transform features: X_transformed = X @ U @ Λ^(-β/2) @ U^T
        transform_matrix = U @ Lambda_beta @ U.T
        X_transformed = X @ transform_matrix
        
        # 5. RBF kernel in transformed space
        K = exp(-||x_i - x_j||^2 / (2σ^2))
        return K
```

**What it does**:
- β=0: Standard RBF kernel (no transformation)
- β>0: Flattens eigenvalue spectrum, upweights low-variance directions
- If signal is in low-variance directions, it leads to improvement
- If signal is in high-variance directions, it leads to no improvement (as expected)

---

### `scripts/synthetic_data_experiment.py`

**Purpose**: Generate synthetic experiment plots

**What it does**:
1. Runs synthetic data evaluation via grid search over β ∈ [0.0, 0.125, 0.25, 0.375, 0.5] and regularization C_values ∈ [0.1, 1.0, 10.0] and creates visualizations
2. Creates 2D synthetic data for explainability in 3 plots:
   - Decision boundaries for β=0 vs β=optimal
   - Performance curve (accuracy vs β)
   - Kernel diagnostics (alignment, condition number)
3. Output: `results/synthetic_data_experiment/plots.png`

---

### `scripts/covariance_structure_test.py`

**Purpose**: Test robustness across different covariance types

**Test matrix** (12 scenarios):
- **Covariance types** (4): Random, Toeplitz, Block-diagonal, Low-rank+noise
- **Signal profiles** (3): Low-variance, High-variance, Power-law decay

**Result**: Performance depends on where class-discriminative signal lives in eigenspace, not just overall conditioning.

---

## Phase 2: Real Data Preparation

### `scripts/download_datasets.py`

**Purpose**: Automated dataset download from Kaggle

**Datasets**:
- CUB-200-2011: 11,788 bird images, 200 species
- Stanford Cars: 16,185 car images, 196 models

**How it works**:
```bash
python3 scripts/download_datasets.py --all
```

1. Uses Kaggle API (`~/.kaggle/kaggle.json` required)
2. Downloads ZIP archives
3. Extracts to `data/`
4. Creates symlinks for consistent paths

---

### `scripts/extract_all_features.py`

**Purpose**: Extract deep features from pretrained models

**Supported models**:
- ResNet50: 2048-D features (final layer before classification)
- ViT-B/16: 768-D features (CLS token)
- ConvNeXt-Base: 1024-D features (global pooling)

**Implementation** (`src/feature_extractor.py`):
```python
def extract_and_save_features(model_name, dataset, output_dir):
    # 1. Load pretrained model
    model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
    model.eval()
    
    # 2. Extract intermediate features (before classifier)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    
    # 3. Batch processing
    for images, labels in dataloader:
        with torch.no_grad():
            features = feature_extractor(images)
    
    # 4. Cache to disk
    np.savez(output_path, features=features, labels=labels)
```

**Usage**:
```bash
# Extract all models for CUB
python3 scripts/extract_all_features.py --dataset cub --all

# Extract specific model
python3 scripts/extract_all_features.py --dataset cub --models resnet50
```

**Output**: `results/features/{dataset}/{model}_{train|test}.npz`

**We do caching because**
- Feature extraction takes ~20 minutes per model
- Reuse features across multiple experiments
- Reproducibility (same features for all runs)

---

## Phase 3: Real Data Evaluation

### `scripts/evaluate_real_data.py`

**Purpose**: Grid search for optimal (β, σ, C) on real data

**Grid search strategy**:
```python
# Parameter grids
beta_values = [0.0, 0.025, 0.05, 0.075, 0.10, 0.125]
C_values = [0.01, 0.05, 0.1, 0.5, 1.0]  # SVM regularization

# For each β:
for beta in beta_values:
    # 1. Compute spectral flattening kernel
    kernel = SpectralFlatteningKernel(beta=beta)
    K = kernel.compute_kernel(X_train)
    
    # 2. Select σ via median heuristic
    sigma = median(pairwise_distances(X_train))
    
    # 3. Grid search over C with 3-fold CV
    for C in C_values:
        cv_scores = []
        for train_fold, val_fold in kfold.split(X_train, y_train):
            svm = SVC(kernel='precomputed', C=C)
            svm.fit(K[train_fold][:, train_fold], y_train[train_fold])
            score = svm.score(K[val_fold][:, train_fold], y_train[val_fold])
            cv_scores.append(score)
        
        # Store per-fold scores for error bars
        results.append({
            'beta': beta,
            'C': C,
            'cv_score': mean(cv_scores),
            'cv_fold_scores': cv_scores,  # For error bars
            'cv_std': std(cv_scores)
        })
    
    # 4. Evaluate best on test set
    test_accuracy = svm.score(K_test, y_test)
```

**Usage**:
```bash
# Evaluate all models for CUB
python3 scripts/evaluate_real_data.py --dataset cub --all

# Single model
python3 scripts/evaluate_real_data.py --dataset cub --model resnet50
```

**Output per model** (in `results/real_data/{dataset}/{model}/`):
1. **results.json**: Full CV fold data for all configurations
2. **results.csv**: Summary table
3. **plots.png**: 4 subplots with error bars:
   - Accuracy vs β (CV ± std, test accuracy)
   - Kernel-Target Alignment vs β
   - Condition Number vs β (log scale)
   - Effective Rank vs β
4. **summary.txt**: Text summary with best parameters

**JSON structure**:
```json
{
  "metadata": {
    "dataset": "CUB-200-2011",
    "model": "resnet50",
    "timestamp": "2026-01-15 14:30:00",
    "cv_folds": 3
  },
  "dataset_stats": {
    "n_train": 5994,
    "n_test": 5794,
    "n_features": 2048,
    "n_classes": 200
  },
  "results": [
    {
      "beta": 0.05,
      "sigma": 45.2,
      "C": 0.1,
      "cv_score": 0.792,
      "cv_fold_scores": [0.789, 0.795, 0.792],
      "cv_std": 0.003,
      "test_accuracy": 0.794,
      "kernel_alignment": 0.342,
      "condition_number": 1234.5,
      "effective_rank": 856
    },
    ...
  ]
}
```

---

## Implementation Details

### 1. Spectral Flattening Mathematics

**Formula**: Given covariance $\Sigma = U \Lambda U^T$:

$$K_\beta(x, x') = \exp\left(-\frac{\|U \Lambda^{-\beta/2} U^T (x - x')\|^2}{2\sigma^2}\right)$$

**Implementation**:
- Regularization: $\Lambda + \epsilon I$ to prevent division by zero
- Numerical stability: Use `np.linalg.eigh` (symmetric eigendecomposition)
- Precomputation: Cache $U \Lambda^{-\beta/2} U^T$ to avoid repeated computation

### 2. Cross-Validation Strategy

- For CUB: ~2000 samples per fold (sufficient for 200 classes)
- For Stanford Cars: ~2700 samples per fold (sufficient for 196 classes)
- Preserves class distribution in each fold via stratification

### 3. Adaptive Bandwidth Selection

**Median heuristic**: $\sigma = \text{median}(\|x_i - x_j\|)$

**Why?**
- Scales with feature distribution
- Avoids manual tuning
- Works well across different models (ResNet, ViT, ConvNeXt have different feature scales)

---

## Reproducibility

All experiments are reproducible by following QUICKSTART.md

**Expected runtime**:
- Synthetic validation: ~5 minutes total
- Feature extraction: ~20-30 minutes per model (one-time)
- Real data evaluation: ~20-30 minutes per model

**Random seeds**: Fixed where applicable (NumPy, scikit-learn CV splits)

---

## Dependencies

See [requirements.txt](requirements.txt) for exact versions.

**Core**:
- NumPy, SciPy (numerical computing)
- scikit-learn (SVM solver)
- PyTorch, torchvision (pretrained models)

**Visualization**:
- matplotlib, seaborn

**Utilities**:
- pandas (data manipulation)
- tqdm (progress bars)
