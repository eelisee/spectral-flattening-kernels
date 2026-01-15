"""
Extract features from pretrained models for benchmark datasets

Supports:
- CUB-200-2011 (200 bird species)
- Stanford Cars (196 car models)

Usage:
    python scripts/extract_all_features.py --dataset cub
    python scripts/extract_all_features.py --dataset stanford_cars
    
Output:
    results/features/{dataset}/{model}_train.npz
    results/features/{dataset}/{model}_test.npz
"""

import os
import sys
import ssl
import torch
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import CUBDataset, StanfordCarsDataset
from src.feature_extractor import extract_and_save_features


DATASET_CONFIG = {
    'cub': {
        'name': 'CUB-200-2011',
        'data_root': 'data/CUB_200_2011',
        'dataset_class': CUBDataset,
        'num_classes': 200,
        'description': '200 bird species'
    },
    'stanford_cars': {
        'name': 'Stanford Cars',
        'data_root': 'data/stanford_cars',
        'dataset_class': StanfordCarsDataset,
        'num_classes': 196,
        'description': '196 car models'
    }
}


def main():
    """Extract features from all models"""
    
    parser = argparse.ArgumentParser(description='Extract features from pretrained models')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['cub', 'stanford_cars'],
                       help='Dataset to extract features from')
    parser.add_argument('--models', type=str, nargs='+',
                       choices=['resnet50', 'vit_b_16', 'convnext_base'],
                       help='Models to use (default: all three models)')
    parser.add_argument('--all', action='store_true',
                       help='Extract features for all models (same as omitting --models)')
    
    args = parser.parse_args()
    
    # Determine which models to use
    if args.all or args.models is None:
        models = ['resnet50', 'vit_b_16', 'convnext_base']
    else:
        models = args.models
    
    # Get dataset config
    config = DATASET_CONFIG[args.dataset]
    data_root = config['data_root']
    dataset_class = config['dataset_class']
    
    # Check if dataset exists
    if not os.path.exists(data_root):
        print(f"Error: Dataset not found at {data_root}")
        sys.exit(1)
    
    print("="*80)
    print(f"FEATURE EXTRACTION: {config['name'].upper()}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print(f"\nDataset: {config['description']}")
    print(f"Location: {data_root}")
    
    # Setup output directory
    output_dir = f"results/features/{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Models to extract features from
    models_config = {
        'resnet50': 'ResNet50 (2048-D)',
        'vit_b_16': 'ViT-B/16 (768-D)',
        'convnext_base': 'ConvNeXt-Base (1024-D)'
    }
    
    models_to_extract = [(m, models_config[m]) for m in models]
    total_models = len(models_to_extract)
    
    for idx, (model_name, description) in enumerate(models_to_extract, 1):
        print(f"\n{'='*80}")
        print(f"MODEL {idx}/{total_models}: {description}")
        print("="*80)
        
        # Training set
        print(f"\n[{model_name}] Extracting training features...")
        train_output = f"{output_dir}/{model_name}_train.npz"
        
        if os.path.exists(train_output):
            print(f"   File exists: {train_output}")
            print(f"   Skipping... (delete file to re-extract)")
        else:
            train_dataset = dataset_class(
                root_dir=data_root,
                train=True,
                transform=None
            )
            
            print(f"   Dataset size: {len(train_dataset)} images")
            print(f"   Extracting features...")
            
            extract_and_save_features(
                dataset=train_dataset,
                model_name=model_name,
                save_path=train_output,
                batch_size=32,
                device=device
            )
        
        # Test set
        print(f"\n[{model_name}] Extracting test features...")
        test_output = f"{output_dir}/{model_name}_test.npz"
        
        if os.path.exists(test_output):
            print(f"   File exists: {test_output}")
            print(f"   Skipping... (delete file to re-extract)")
        else:
            test_dataset = dataset_class(
                root_dir=data_root,
                train=False,
                transform=None
            )
            
            print(f"   Dataset size: {len(test_dataset)} images")
            print(f"   Extracting features...")
            
            extract_and_save_features(
                dataset=test_dataset,
                model_name=model_name,
                save_path=test_output,
                batch_size=32,
                device=device
            )
    
    # Summary
    print("\n" + "="*80)
    print("EXTRACTION SUMMARY")
    print("="*80)
    
    print(f"\nDataset: {config['name']}")
    print(f"Extracted features:")
    
    for model_name in args.models:
        train_path = f"{output_dir}/{model_name}_train.npz"
        test_path = f"{output_dir}/{model_name}_test.npz"
        
        if os.path.exists(train_path) and os.path.exists(test_path):
            train_data = np.load(train_path)
            test_data = np.load(test_path)
            
            print(f"\n{model_name}:")
            print(f"    Train: {train_data['features'].shape}")
            print(f"    Test:  {test_data['features'].shape}")
            print(f"    Files:")
            print(f"      - {train_path}")
            print(f"      - {test_path}")
        else:
            print(f"\n {model_name}: INCOMPLETE")
    
    print("\n" + "="*80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == '__main__':
    main()
