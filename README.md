# Spectral Flattening for Kernel Methods on Pretrained Features

**Advanced Machine Learning Project**  
at ENSAE - École nationale de la statistique et de l’administration économique - IP Paris 

## Project Overview

This project investigates **spectral flattening** as a preprocessing technique for kernel SVM classification on deep neural network features. We analyze the theoretical foundations and empirical performance of transforming pretrained features via eigenvalue manipulation to improve classification accuracy on fine-grained visual recognition tasks.

### Research Question

Can spectral flattening of covariance structure improve kernel SVM performance on pretrained CNN/ViT features, and under what conditions does this improvement occur?

### Key Findings

- **Synthetic validation**: Spectral flattening improves classification when signal components align with low-variance feature directions (validated across 12+ test scenarios)
- **Real data results**: Pretrained features show limited improvement, suggesting signal already aligns with high-variance components (CUB-200-2011, Stanford Cars datasets)
- **Theoretical insight**: Performance depends on spectral profile of class-discriminative information, not just overall conditioning

## Repository Structure

```
advancedml/
├── README.md                 # This file
├── RUN_EXPERIMENTS.md        # Detailed execution guide
├── CODE_DOCUMENTATION.md     # Code attribution & originality
├── requirements.txt          # Python dependencies
├── paper.tex                 # Project report (NeurIPS format)
│
├── src/                      # Core implementation
│   ├── __init__.py
│   ├── data_loader.py        # Dataset loading (CUB, Stanford Cars)
│   ├── feature_extractor.py  # Deep feature extraction
│   ├── kernels.py            # Spectral flattening kernel
│   ├── svm_utils.py          # SVM training with CV
│   └── visualization.py      # Publication plots
│
├── scripts/                  # Experiment scripts
│   ├── synthetic_data_test.py           # Phase 1: Theory validation
│   ├── synthetic_data_experiment.py     # Phase 1: Synthetic plots
│   ├── grid_search_test.py.py           # Phase 1: Gridsearch on synthetic data
│   ├── covariance_structure_test.py     # Phase 1: Robustness tests
│   ├── download_datasets.py             # Phase 2: Data download
│   ├── extract_all_features.py          # Phase 2: Feature extraction
│   └── evaluate_real_data.py            # Phase 3: Real data evaluation
│
├── data/                     # Datasets (downloaded)
│   ├── CUB_200_2011/
│   └── stanford_cars/
│
└── results/                  # Generated outputs
    ├── synthetic_data_test/
    ├── synthetic_data_experiment/
    ├── covariance_structure_test/
    ├── features/             # Cached features
    └── real_data/            # Per-model results
        └── {dataset}/{model}/
            ├── results.json  # Full CV fold data
            ├── results.csv
            ├── plots.png     # 4 subplots with error bars
            └── summary.txt
```

## Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/eelisee/advancedml.git
cd advancedml

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Synthetic Validation

```bash
# Comprehensive theory validation
python3 scripts/synthetic_data_test.py

# Generate synthetic experiment plots
python3 scripts/synthetic_data_experiment.py

# Test covariance structure sensitivity
python3 scripts/covariance_structure_test.py
```

**Expected output**: All validation tests PASS, plots in `results/synthetic_*/`

### 3. Download Real Datasets

```bash
python3 scripts/download_datasets.py --all
```

**Note**: Requires Kaggle API credentials for CUB-200-2011 and Stanford Cars datasets

### 4. Extract Deep Features

```bash
python3 scripts/extract_all_features.py
```

**Output**: Cached features in `results/features/` (~2GB disk space)

### 5. Run Real Data Evaluation

```bash
# Single model/dataset
python3 scripts/evaluate_real_data.py --dataset CUB-200-2011 --model resnet50

# All models for CUB
python3 scripts/evaluate_real_data.py --dataset CUB-200-2011 --model resnet50
python3 scripts/evaluate_real_data.py --dataset CUB-200-2011 --model vit_b_16
python3 scripts/evaluate_real_data.py --dataset CUB-200-2011 --model convnext_base
```

**Output per model**: JSON with CV folds, CSV summary, combined plot with 4 subplots (all with error bars)

### Complete Execution Guide

See [RUN_EXPERIMENTS.md](RUN_EXPERIMENTS.md) for detailed step-by-step instructions, expected outputs, and troubleshooting.

## Method Overview

### Spectral Flattening Kernel

Given feature covariance $\Sigma = U \Lambda U^T$ (eigendecomposition), we apply:

$$K_\beta(x, x') = \exp\left(-\frac{\|U \Lambda^{-\beta/2} U^T (x - x')\|^2}{2\sigma^2}\right)$$

where:
- $\beta \in [0, 1]$ controls flattening strength (0 = standard RBF, 1 = whitening)
- $\sigma$ is bandwidth (selected via median heuristic)
- Regularization $\Lambda + \epsilon I$ prevents numerical issues

### Key Hyperparameters

- **β (beta)**: Spectral flattening strength  
  Grid: [0.0, 0.025, 0.05, 0.075, 0.10, 0.125]
  
- **σ (sigma)**: RBF bandwidth  
  Adaptive median heuristic: $\sigma = \text{median}(\|x_i - x_j\|)$
  
- **C**: SVM regularization  
  Grid: [0.01, 0.05, 0.1, 0.5, 1.0]  
  Selected via 3-fold cross-validation

### Performance Metrics

- **Test Accuracy**: Final evaluation on held-out test set
- **CV Score**: Mean 3-fold cross-validation accuracy (with std)
- **Kernel-Target Alignment**: Similarity between kernel matrix and ideal target kernel
- **Condition Number**: $\kappa(\tilde{K}) = \lambda_{\max} / \lambda_{\min}$
- **Effective Rank**: Number of eigenvalues explaining 90% variance

## Datasets

### CUB-200-2011 (Caltech-UCSD Birds)
- **Task**: 200-way bird species classification
- **Split**: 5,994 train / 5,794 test images
- **Difficulty**: Fine-grained visual recognition
- **Source**: [Caltech Birds Dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)

### Stanford Cars
- **Task**: 196-way car model classification
- **Split**: 8,144 train / 8,041 test images
- **Difficulty**: Fine-grained recognition (viewpoint/lighting variation)
- **Source**: [Stanford Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

## Deep Feature Extractors

All models use ImageNet-pretrained weights via `torchvision.models`:

1. **ResNet50**: 2048-D features from final layer (pre-classification)
2. **ViT-B/16**: 768-D features from CLS token
3. **ConvNeXt-Base**: 1024-D features from global pooling

Features extracted once and cached for reuse.

## Key Results

### Synthetic Data (Theory Validation)
**All 4 test scenarios PASS**:
- Low-variance signal: +15-25% improvement with optimal β
- High-variance signal: ~0% improvement (as expected)
- Anisotropy sensitivity: Improvement scales with κ
- Sample size robustness: Consistent across 25-200 samples/class

### Real Data (CUB-200-2011)
- **ResNet50**: 59.65% → 59.65% (+0.0pp, β=0.0 optimal) - No improvement, degradation with β>0
- **ViT-B/16**: 63.34% → 63.34% (+0.0pp, β=0.0 optimal) - No improvement, degradation with β>0
- **ConvNeXt-Base**: 61.53% → 63.13% (+1.6pp with β=0.125) - **Only model showing improvement**

### Real Data (Stanford Cars)
- **ResNet50**: 0.61% → 11.86% (+11.2pp with β=0.125) - Catastrophic baseline, extreme overfitting
- **ViT-B/16**: 0.66% → 0.90% (+0.2pp with β=0.125) - Catastrophic baseline, minimal improvement
- **ConvNeXt-Base**: 1.44% → 7.67% (+6.2pp with β=0.125) - Best absolute performance, still poor

**Interpretation**: Pretrained features already align signal with high-variance components, limiting flattening benefits.

## Dependencies

- Python 3.8+
- PyTorch 2.0+ (CPU sufficient, GPU optional)
- scikit-learn 1.3+
- NumPy, SciPy, pandas
- matplotlib, seaborn (visualization)

See [requirements.txt](requirements.txt) for complete list.

## Citation & Acknowledgments

This project builds upon theoretical foundations from:
- Kernel Methods in Machine Learning (Hofmann et al., 2008)
- Spectral Regularization for Support Vector Machines (various works)
- Fine-Grained Visual Categorization literature

For code attribution and original contributions, see [CODE_DOCUMENTATION.md](CODE_DOCUMENTATION.md).

## License

Educational project for ENSAE Advanced Machine Learning course (January 2026).

## Contact

Questions about this project? Contact via GitHub issues or email.
