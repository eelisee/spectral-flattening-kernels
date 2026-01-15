"""
Download benchmark datasets for spectral flattening experiments.

This script downloads CUB-200-2011 and Stanford Cars datasets via kagglehub
and creates symlinks in the data/ directory for consistent access.

Usage:
    # Download both datasets
    python scripts/download_datasets.py --all
    
    # Download specific dataset
    python scripts/download_datasets.py --dataset cub
    python scripts/download_datasets.py --dataset stanford_cars
"""

import argparse
import os
from pathlib import Path
import kagglehub


def download_cub_dataset(data_dir: Path) -> Path:
    """
    Download CUB-200-2011 dataset via kagglehub.
    
    Dataset: Caltech-UCSD Birds-200-2011
    - 200 bird species
    - 11,788 images (5,994 train, 5,794 test)
    
    Args:
        data_dir: Root data directory for creating symlink
        
    Returns:
        Path to downloaded dataset
    """
    print("=" * 80)
    print("DOWNLOADING CUB-200-2011 DATASET")
    print("=" * 80)
    print("Dataset: Caltech-UCSD Birds-200-2011")
    print("Source: kagglehub (wenewone/cub2002011)")
    print("Size: ~2 GB")
    print()
    
    # Download via kagglehub
    download_path = kagglehub.dataset_download("wenewone/cub2002011")
    print(f"Downloaded to: {download_path}")
    
    # Create symlink in data/
    symlink_path = data_dir / "CUB_200_2011"
    if symlink_path.exists() or symlink_path.is_symlink():
        print(f"Symlink already exists: {symlink_path}")
        if symlink_path.is_symlink():
            print(f"  Points to: {os.readlink(symlink_path)}")
    else:
        os.symlink(download_path, symlink_path)
        print(f"Created symlink: {symlink_path} -> {download_path}")
    
    print()
    return Path(download_path)


def download_stanford_cars_dataset(data_dir: Path) -> Path:
    """
    Download Stanford Cars dataset via kagglehub.
    
    Dataset: Stanford Cars
    - 196 car models (make/model/year combinations)
    - 16,185 images (8,144 train, 8,041 test)
    
    Args:
        data_dir: Root data directory for creating symlink
        
    Returns:
        Path to downloaded dataset
    """
    print("=" * 80)
    print("DOWNLOADING STANFORD CARS DATASET")
    print("=" * 80)
    print("Dataset: Stanford Cars")
    print("Source: kagglehub (eduardo4jesus/stanford-cars-dataset)")
    print("Size: ~2 GB")
    print()
    
    # Download via kagglehub
    download_path = kagglehub.dataset_download("eduardo4jesus/stanford-cars-dataset")
    print(f"Downloaded to: {download_path}")
    
    # Create symlink in data/
    symlink_path = data_dir / "stanford_cars"
    if symlink_path.exists() or symlink_path.is_symlink():
        print(f"Symlink already exists: {symlink_path}")
        if symlink_path.is_symlink():
            print(f"  Points to: {os.readlink(symlink_path)}")
    else:
        os.symlink(download_path, symlink_path)
        print(f"Created symlink: {symlink_path} -> {download_path}")
    
    print()
    return Path(download_path)


def main():
    parser = argparse.ArgumentParser(
        description="Download benchmark datasets for spectral flattening experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                # Download both datasets
                python scripts/download_datasets.py --all
                
                # Download only CUB-200-2011
                python scripts/download_datasets.py --dataset cub
                
                # Download only Stanford Cars
                python scripts/download_datasets.py --dataset stanford_cars

                Dataset Information:
                CUB-200-2011:    200 bird species, 11,788 images (~2 GB)
                Stanford Cars:   196 car models, 16,185 images (~2 GB)
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['cub', 'stanford_cars'],
        help='Specific dataset to download (cub or stanford_cars)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all datasets'
    )
    
    args = parser.parse_args()
    
    # Ensure data directory exists
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Validate arguments
    if not args.all and not args.dataset:
        parser.error("Must specify either --dataset or --all")
    
    # Download requested datasets
    if args.all or args.dataset == 'cub':
        download_cub_dataset(data_dir)
    
    if args.all or args.dataset == 'stanford_cars':
        download_stanford_cars_dataset(data_dir)
    
    print("=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)
    print(f"Data directory: {data_dir}")

if __name__ == "__main__":
    main()
