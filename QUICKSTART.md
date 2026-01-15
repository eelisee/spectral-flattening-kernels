# Experimental Pipeline - Step-by-Step Execution Guide

Complete instructions to reproduce all results, plots, and tables.

---

## Prerequisites

Ensure virtual environment is activated:

```bash
source .venv/bin/activate
```

---

## Phase 1: Synthetic Data Validation

### Step 1.1: Comprehensive Theory Validation

```bash
python3 scripts/synthetic_data_test.py
```

**What it does:**
- Tests LOW-variance vs HIGH-variance signal scenarios
- Validates signal strength sensitivity (1.0 to 5.0)
- Tests anisotropy sensitivity (κ = 10 to 10,000)
- Validates sample size robustness (25 to 200 samples/class)
- Tests realistic CUB-like scenarios

**Output:**
- `results/synthetic_data_test/validation_results.csv`
- Terminal output showing PASS/FAIL for each test

---

### Step 1.2: Generate Synthetic Plots for Paper

```bash
python3 scripts/synthetic_data_experiment.py
```

**What it does:**
- Creates publication-ready synthetic data experiment
- Demonstrates spectral flattening on controlled data
- Generates visualizations and summary statistics

**Output:**
- `results/synthetic_data_experiment/results.csv`
- `results/synthetic_data_experiment/plots.png`
- `results/synthetic_data_experiment/decision_boundaries.png`
- `results/synthetic_data_experiment/experiment_summary.txt`

---

### Step 1.3: Validate Grid Search Implementation

```bash
python3 scripts/grid_search_test.py
```

**What it does:**
- Tests grid search on synthetic data
- Validates parameter selection via cross-validation
- Confirms grid search finds optimal β

**Output:**
- Terminal output with grid search results

---

### Step 1.4: Test Covariance Structure Effects

```bash
python3 scripts/covariance_structure_test.py
```

**What it does:**
- Tests spectral flattening under realistic covariance structures
- Validates that Δ = U α construction (in eigenspace) is critical
- Tests 4 covariance types × 3 signal profiles = 12 scenarios
  - Covariance: random, toeplitz, block, lowrank+noise
  - Signal: low-variance, high-variance, power-law
- Confirms theory: performance depends on spectral profile of signal

**Output:**
- `results/covariance_structure_test/results.csv`
- `results/covariance_structure_test/performance_grid.png`
- `results/covariance_structure_test/improvement_heatmap.png`
- `results/covariance_structure_test/summary.txt`

---

## Phase 2: Real Data Preparation

### Step 2.1: Download Datasets

```bash
python3 scripts/download_datasets.py --all
```

**What it does:**
- Downloads CUB-200-2011 from Kaggle
- Downloads Stanford Cars from Kaggle (optional)
- Creates symlinks in `data/` directory

**Output:**
- `data/CUB_200_2011/` (11,788 images)
- `data/stanford_cars/` (16,185 images) - if included

**Note:** Skip if datasets already downloaded

---

### Step 2.2: Extract Deep Features

```bash
python3 scripts/extract_all_features.py
```

**What it does:**
- Extracts ResNet50 features (2048-D)
- Extracts ViT-B/16 features (768-D)
- Extracts ConvNeXt-Base features (1024-D)
- Caches features to disk for reuse

**Output:**
- `results/features/CUB_ResNet50_train.npy`
- `results/features/CUB_ResNet50_test.npy`
- `results/features/CUB_ViT_B16_train.npy`
- `results/features/CUB_ViT_B16_test.npy`
- `results/features/CUB_ConvNeXt_Base_train.npy`
- `results/features/CUB_ConvNeXt_Base_test.npy`

**Expected runtime:** 
- First run: ~20-30 minutes (extracts all features)
- Subsequent runs: <1 second (loads cached features)

**Note:** Requires ~2GB disk space for cached features

---

## Phase 3: Real Data Evaluation

### Step 3.1: Run Grid Search on Real Data

```bash
python3 scripts/evaluate_real_data.py
```

**What it does:**
- Runs spectral flattening grid search for each architecture
- Tests β ∈ [0.0, 0.025, 0.05, 0.075, 0.10, 0.125]
- Tests σ via adaptive median heuristic
- Tests C ∈ [0.01, 0.05, 0.1, 0.5, 1.0]
- Uses 3-fold cross-validation for selection
- Computes kernel diagnostics (alignment, condition number, etc.)
- Generates plots with error bars

**Output per model:**
- `results/real_data/{dataset}/{model}/results.json` - Full CV fold data
- `results/real_data/{dataset}/{model}/results.csv` - Summary table
- `results/real_data/{dataset}/{model}/plots.png` - 4 plots with error bars:
  1. Accuracy vs β (CV score with error bars + test accuracy)
  2. Kernel-Target Alignment vs β
  3. Condition Number vs β (log scale)
  4. Effective Rank vs β
- `results/real_data/{dataset}/{model}/summary.txt` - Text summary

**Expected runtime:**
- ResNet50: ~20 minutes
- ViT-B/16: ~15 minutes
- ConvNeXt-Base: ~20 minutes

**Note:** Can be interrupted and resumed per architecture

---

## Summary of Generated Files

After completing all steps:

**Synthetic Data:**
- 1 validation CSV
- 3 plots
- 1 summary text file

**Real Data (per model/dataset):**
- 1 results JSON with full CV fold data
- 1 results CSV
- 1 combined plot with 4 subplots (all with error bars)
- 1 summary text file

---

## Quick Check Commands

Verify synthetic validation passed:
```bash
cat results/synthetic_data_test/validation_results.csv | grep FAIL
```
*Expected: Empty output (all tests PASS)*

Check real data results exist:
```bash
ls -lh results/real_data/CUB-200-2011/*/results.json
```
*Expected: JSON files for each evaluated model*

Count generated plots:
```bash
ls results/synthetic_data_experiment/*.png results/covariance_structure_test/*.png results/real_data/**/plots.png | wc -l
```
*Expected: 5+ plots (synthetic) + 1 per evaluated model*

---

## Troubleshooting

**If synthetic_data_test.py shows FAIL:**
- Check if HIGH-variance tests show near-zero improvement (expected)
- LOW-variance tests should show positive improvement

**If extract_all_features.py fails:**
- Ensure ~2GB disk space available
- Check torch/torchvision installation
- GPU not required but speeds up extraction

**If evaluate_real_data.py is slow:**
- Normal: ~1-2 minutes per β value
- Total ~30-60 minutes for all architectures
- Can run models separately with `--model` flag

**If evaluate_real_data.py shows errors:**
- Ensure extract_all_features.py completed successfully
- Check that feature files exist in results/features/

---

## Next Steps

After all experiments complete:
1. Review plots in `results/real_data/{dataset}/{model}/`
2. Check JSON/CSV files for detailed results
3. Update paper with findings
4. Use synthetic validation to argue theoretical validity
5. Use real data results to discuss pretrained feature limitations

---
