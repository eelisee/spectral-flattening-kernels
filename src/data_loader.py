"""
Data loading utilities for CUB-200-2011 and Stanford Cars datasets
"""

import os
import numpy as np
from PIL import Image
from typing import Tuple, List, Dict
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat


class CUBDataset(Dataset):
    """CUB-200-2011 Bird Dataset"""
    
    def __init__(self, root_dir: str, train: bool = True, transform=None):
        """
        Args:
            root_dir: Path to CUB_200_2011 directory
            train: If True, load training set, otherwise test set
            transform: Optional transform to be applied on images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        
        # Load image paths
        images_file = os.path.join(root_dir, 'images.txt')
        with open(images_file, 'r') as f:
            self.images = [line.strip().split(' ', 1)[1] for line in f]
        
        # Load labels
        labels_file = os.path.join(root_dir, 'image_class_labels.txt')
        with open(labels_file, 'r') as f:
            self.labels = [int(line.strip().split()[1]) - 1 for line in f]  # 0-indexed
        
        # Load train/test split
        split_file = os.path.join(root_dir, 'train_test_split.txt')
        with open(split_file, 'r') as f:
            is_train = [int(line.strip().split()[1]) for line in f]
        
        # Filter based on train/test
        indices = [i for i, flag in enumerate(is_train) if flag == (1 if train else 0)]
        self.images = [self.images[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        
        # Load class names
        classes_file = os.path.join(root_dir, 'classes.txt')
        with open(classes_file, 'r') as f:
            self.class_names = [line.strip().split(' ', 1)[1] for line in f]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, 'images', self.images[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_name(self, label: int) -> str:
        """Get class name for a given label"""
        return self.class_names[label]


def load_cub_metadata(root_dir: str) -> Dict:
    """Load metadata about the CUB dataset"""
    
    # Count classes
    classes_file = os.path.join(root_dir, 'classes.txt')
    with open(classes_file, 'r') as f:
        num_classes = len(f.readlines())
    
    # Count train/test samples
    split_file = os.path.join(root_dir, 'train_test_split.txt')
    with open(split_file, 'r') as f:
        splits = [int(line.strip().split()[1]) for line in f]
    
    num_train = sum(splits)
    num_test = len(splits) - num_train
    
    return {
        'num_classes': num_classes,
        'num_train': num_train,
        'num_test': num_test,
        'total_images': len(splits)
    }


class StanfordCarsDataset(Dataset):
    """Stanford Cars Dataset (196 classes)"""
    
    def __init__(self, root_dir: str, train: bool = True, transform=None):
        """
        Args:
            root_dir: Path to stanford_cars directory
            train: If True, load training set, otherwise test set
            transform: Optional transform to be applied on images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        
        # Load annotations
        devkit_dir = os.path.join(root_dir, 'car_devkit', 'devkit')
        
        if train:
            annos_file = os.path.join(devkit_dir, 'cars_train_annos.mat')
            img_dir = os.path.join(root_dir, 'cars_train', 'cars_train')
        else:
            annos_file = os.path.join(devkit_dir, 'cars_test_annos.mat')
            img_dir = os.path.join(root_dir, 'cars_test', 'cars_test')
        
        # Load .mat file
        annos = loadmat(annos_file)
        annotations = annos['annotations'][0]
        
        self.images = []
        self.labels = []
        
        for anno in annotations:
            # Extract image name and class (bbox info also available if needed)
            fname = str(anno['fname'][0])
            self.images.append(os.path.join(img_dir, fname))
            
            # class is 1-indexed in .mat file, convert to 0-indexed
            if train:
                class_id = int(anno['class'][0][0]) - 1
                self.labels.append(class_id)
            else:
                # Test set might not have labels in some versions
                try:
                    class_id = int(anno['class'][0][0]) - 1
                    self.labels.append(class_id)
                except:
                    self.labels.append(0)  # Placeholder
        
        # Load class names
        meta_file = os.path.join(devkit_dir, 'cars_meta.mat')
        meta = loadmat(meta_file)
        self.class_names = [str(c[0]) for c in meta['class_names'][0]]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_name(self, label: int) -> str:
        """Get class name for a given label"""
        return self.class_names[label]
