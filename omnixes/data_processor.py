import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pydantic import BaseModel

from omnixas.data.ml_data import MLData, MLSplits
from omnixas.featurizer.m3gnet_featurizer import M3GNetSiteFeaturizer
from omnixes.xes_data import XESSpectrum, find_site_index_for_element


def featurize_xes_data(
    xes_data_list: List[XESSpectrum], 
    featurizer: M3GNetSiteFeaturizer
) -> MLData:
    """
    Process a list of XESSpectrum objects into ML-ready data.
    
    Args:
        xes_data_list: List of XESSpectrum objects
        featurizer: An initialized M3GNetSiteFeaturizer
        
    Returns:
        MLData: Container with features (X) and labels (y)
    """
    if not xes_data_list:
        return MLData(X=np.array([]), y=np.array([]))
    
    X_list = []
    y_list = []
    
    for xes in xes_data_list:
        # Find the site matching the target element
        site_index = find_site_index_for_element(xes.structure, xes.element)
        
        # Extract features for that site
        features = featurizer.featurize(xes.structure, site_index)
        
        # Use the intensities as the target values
        X_list.append(features)
        y_list.append(xes.intensities)
    
    # Stack all samples into arrays
    X = np.stack(X_list, axis=0)  # shape (n_samples, feature_dim)
    y = np.stack(y_list, axis=0)  # shape (n_samples, spectrum_length)
    
    return MLData(X=X, y=y)


def split_data(
    full_data: List[XESSpectrum],
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    seed: int = 42
) -> Tuple[List[XESSpectrum], List[XESSpectrum], List[XESSpectrum]]:
    """
    Split the data into train/val/test sets.
    
    Args:
        full_data: List of XESSpectrum objects
        train_fraction: Fraction for training set
        val_fraction: Fraction for validation set
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_list, val_list, test_list)
    """
    # Make a copy to avoid modifying original
    shuffled_data = full_data.copy()
    
    # Set seed for reproducibility
    random.seed(seed)
    random.shuffle(shuffled_data)
    
    n_total = len(shuffled_data)
    n_train = int(n_total * train_fraction)
    n_val = int(n_total * val_fraction)
    
    train_data = shuffled_data[:n_train]
    val_data = shuffled_data[n_train:n_train+n_val]
    test_data = shuffled_data[n_train+n_val:]
    
    return train_data, val_data, test_data


def create_ml_splits(
    train_xes: List[XESSpectrum],
    val_xes: List[XESSpectrum],
    test_xes: List[XESSpectrum],
    featurizer: M3GNetSiteFeaturizer
) -> MLSplits:
    """
    Create ML splits from XES data.
    
    Args:
        train_xes: Training set XES data
        val_xes: Validation set XES data
        test_xes: Test set XES data
        featurizer: An initialized M3GNetSiteFeaturizer
        
    Returns:
        MLSplits: Container with train, val, and test MLData
    """
    # Featurize each split
    train_data = featurize_xes_data(train_xes, featurizer)
    val_data = featurize_xes_data(val_xes, featurizer)
    test_data = featurize_xes_data(test_xes, featurizer)
    
    # Create the MLSplits object
    return MLSplits(train=train_data, val=val_data, test=test_data)


def process_element_data(
    element_data: List[XESSpectrum],
    featurizer: M3GNetSiteFeaturizer,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    seed: int = 42
) -> MLSplits:
    """
    Process all data for a single element into ML splits.
    
    Args:
        element_data: List of XESSpectrum objects for one element
        featurizer: An initialized M3GNetSiteFeaturizer
        train_fraction: Fraction for training
        val_fraction: Fraction for validation
        seed: Random seed
        
    Returns:
        MLSplits object with train/val/test data
    """
    # Split the data
    train_xes, val_xes, test_xes = split_data(
        element_data,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        seed=seed
    )
    
    # Create MLSplits
    return create_ml_splits(train_xes, val_xes, test_xes, featurizer)


class ElementDataset(BaseModel):
    """Container for processed element data and splits."""
    raw_data: List[XESSpectrum]
    ml_splits: MLSplits
    element: str
    
    class Config:
        arbitrary_types_allowed = True 