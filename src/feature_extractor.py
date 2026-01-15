"""
Feature extraction using pretrained models (ResNet50, ViT)
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple


class FeatureExtractor:
    """Extract features from pretrained models"""
    
    def __init__(self, model_name: str = 'resnet50', device: str = 'cpu'):
        """
        Args:
            model_name: Name of pretrained model ('resnet50', 'vit_b_16', 'convnext_base')
            device: Device to run model on
        """
        self.model_name = model_name
        self.device = device
        self.model = self._load_model()
        self.transform = self._get_transform()
    
    def _load_model(self) -> nn.Module:
        """Load pretrained model and modify for feature extraction"""
        
        if self.model_name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            # Remove the final classification layer
            model = nn.Sequential(*list(model.children())[:-1])
            
        elif self.model_name == 'vit_b_16':
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            # Remove classification head, keep encoder
            model.heads = nn.Identity()
            
        elif self.model_name == 'convnext_base':
            model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
            # Remove classifier
            model.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def _get_transform(self):
        """Get appropriate preprocessing transform for the model"""
        
        if self.model_name in ['resnet50', 'convnext_base']:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        elif self.model_name == 'vit_b_16':
            return transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def extract_features(self, dataset, batch_size: int = 32, 
                        num_workers: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from dataset
        
        Args:
            dataset: Dataset to extract features from
            batch_size: Batch size for extraction
            num_workers: Number of workers for data loading
            
        Returns:
            features: (N, D) array of features
            labels: (N,) array of labels
        """
        # Apply our transform to the dataset if it doesn't have one
        if dataset.transform is None:
            dataset.transform = self.transform
        
        dataloader = DataLoader(dataset, batch_size=batch_size, 
                               shuffle=False, num_workers=num_workers)
        
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc=f"Extracting {self.model_name} features"):
                images = images.to(self.device)
                features = self.model(images)
                
                # Flatten features
                features = features.view(features.size(0), -1)
                
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.numpy())
        
        features = np.vstack(all_features)
        labels = np.concatenate(all_labels)
        
        return features, labels


def extract_and_save_features(dataset, model_name: str, save_path: str, 
                              batch_size: int = 32, device: str = 'cpu'):
    """
    Extract features and save to disk
    
    Args:
        dataset: Dataset to extract features from
        model_name: Name of pretrained model
        save_path: Path to save features (as .npz file)
        batch_size: Batch size for extraction
        device: Device to run model on
    """
    extractor = FeatureExtractor(model_name, device)
    features, labels = extractor.extract_features(dataset, batch_size)
    
    np.savez(save_path, features=features, labels=labels)
    print(f"Features saved to {save_path}")
    print(f"Feature shape: {features.shape}")
    
    return features, labels
