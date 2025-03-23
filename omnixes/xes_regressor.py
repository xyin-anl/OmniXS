import torch
import torch.nn as nn
from typing import List, Optional, Dict
import numpy as np
import lightning

from omnixas.model.xasblock import XASBlock
from omnixas.model.xasblock_regressor import XASBlockRegressor
from omnixas.data.ml_data import MLSplits


class ElementEmbeddingLightningModule(lightning.LightningModule):
    """
    Lightning module for element embedding models.
    
    This module handles the training, validation, and prediction loops
    for models that use element embeddings alongside structural features.
    """
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss = torch.nn.MSELoss()
        
    def forward(self, x, element_ids):
        return self.model(x, element_ids)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
        
    def logged_loss(self, name, y, y_pred):
        loss = self.loss(y, y_pred)
        self.log(name, loss, on_step=False, on_epoch=True)
        return loss
        
    def training_step(self, batch, batch_idx):
        x, y, element_ids = batch
        y_pred = self.model(x, element_ids)
        return self.logged_loss("train_loss", y, y_pred)
        
    def validation_step(self, batch, batch_idx):
        x, y, element_ids = batch
        y_pred = self.model(x, element_ids)
        return self.logged_loss("val_loss", y, y_pred)
        
    def test_step(self, batch, batch_idx):
        x, y, element_ids = batch
        y_pred = self.model(x, element_ids)
        return self.logged_loss("test_loss", y, y_pred)
        
    def predict_step(self, batch, batch_idx):
        x, y, element_ids = batch
        return self.model(x, element_ids)


class XESBlockRegressor(XASBlockRegressor):
    """
    Regressor for predicting XES spectra.
    
    This class extends XASBlockRegressor to properly handle XES data.
    The main difference is that the output dimension is typically different
    for XES spectra compared to XAS spectra.
    
    The model uses the same underlying XASBlock neural network architecture
    but adjusts the output dimension to match XES spectra length.
    """
    
    def __init__(
        self,
        input_dim: int = 64,
        output_dim: int = 360,  # Typical length of XES spectra
        hidden_dims: List[int] = [200, 200],
        max_epochs: int = 1000,
        initial_lr: float = 1e-3,
        batch_size: int = 32,
        directory: str = "xes_checkpoints",
        **kwargs
    ):
        """
        Initialize the XES regressor.
        
        Args:
            input_dim: Dimension of input features (typically 64 for M3GNet)
            output_dim: Dimension of output XES spectra (e.g., 360 points)
            hidden_dims: List of hidden layer dimensions
            max_epochs: Maximum training epochs
            initial_lr: Initial learning rate
            batch_size: Batch size for training
            directory: Directory to save model checkpoints
            **kwargs: Additional arguments passed to XASBlockRegressor
        """
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims, 
            max_epochs=max_epochs,
            initial_lr=initial_lr,
            batch_size=batch_size,
            directory=directory,
            **kwargs
        )


class ElementEmbeddingModule(nn.Module):
    """
    Module that combines element embeddings with structural features.
    
    This module adds an embedding layer for element types and concatenates
    the embeddings with the structural features (M3GNet) before feeding
    them into the XASBlock.
    """
    
    def __init__(
        self,
        num_elements: int,
        embedding_dim: int = 16,
        feature_dim: int = 64,
        hidden_dims: List[int] = [300, 300, 300],
        output_dim: int = 360
    ):
        """
        Initialize the element embedding module.
        
        Args:
            num_elements: Number of different elements to embed
            embedding_dim: Dimension of element embeddings
            feature_dim: Dimension of M3GNet features
            hidden_dims: Hidden layer dimensions
            output_dim: Output spectrum dimension
        """
        super().__init__()
        
        # Create embedding layer for element types
        self.element_embedding = nn.Embedding(num_elements, embedding_dim)
        
        # Create XASBlock for the combined input
        combined_input_dim = feature_dim + embedding_dim
        self.xas_block = XASBlock(
            input_dim=combined_input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim
        )
        
    def forward(self, x, element_ids):
        """
        Forward pass with features and element IDs.
        
        Args:
            x: Structural features (B, feature_dim)
            element_ids: Element IDs for each sample (B, 1)
            
        Returns:
            Model predictions (B, output_dim)
        """
        # Get element embeddings
        element_embeddings = self.element_embedding(element_ids.squeeze(-1))  # (B, embedding_dim)
        
        # Concatenate features and embeddings
        combined_features = torch.cat([x, element_embeddings], dim=1)  # (B, feature_dim + embedding_dim)
        
        # Pass through XASBlock
        return self.xas_block(combined_features)


class UniversalXESBlockRegressor(XASBlockRegressor):
    """
    Universal regressor for predicting XES spectra of any element.
    
    This class extends XASBlockRegressor to handle data from multiple elements.
    It incorporates element information through an embedding layer that is
    concatenated with the structural features.
    """
    
    def __init__(
        self,
        num_elements: int,
        element_to_id: Dict[str, int],
        input_dim: int = 64,
        embedding_dim: int = 16,
        output_dim: int = 360,
        hidden_dims: List[int] = [300, 300, 300],
        max_epochs: int = 1000,
        initial_lr: float = 1e-3,
        batch_size: int = 32,
        directory: str = "universal_xes_checkpoints",
        **kwargs
    ):
        """
        Initialize the universal XES regressor.
        
        Args:
            num_elements: Number of different elements to embed
            element_to_id: Mapping from element symbols to integer IDs
            input_dim: Dimension of input features (typically 64 for M3GNet)
            embedding_dim: Dimension of element embeddings
            output_dim: Dimension of output XES spectra
            hidden_dims: Hidden layer dimensions
            max_epochs: Maximum training epochs
            initial_lr: Initial learning rate
            batch_size: Batch size for training
            directory: Directory to save model checkpoints
            **kwargs: Additional arguments
        """
        # Initialize parent class with dummy XASBlock (will be replaced)
        super(XASBlockRegressor, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            max_epochs=max_epochs,
            initial_lr=initial_lr,
            batch_size=batch_size,
            directory=directory,
            **kwargs
        )
        
        # Store element mapping
        self.element_to_id = element_to_id
        self.num_elements = num_elements
        self.embedding_dim = embedding_dim
        
        # Create model with element embedding
        self.model.model = ElementEmbeddingModule(
            num_elements=num_elements,
            embedding_dim=embedding_dim,
            feature_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim
        )
    
    def fit(self, ml_split: MLSplits):
        """
        Train the model on data from multiple elements.
        
        Args:
            ml_split: MLSplits containing combined data from all elements,
                     with element_ids in metadata
                     
        Returns:
            Self for chaining
        """
        from torch.utils.data import TensorDataset, DataLoader
        from lightning import Trainer
        
        # Extract element IDs from metadata
        train_element_ids = ml_split.train.metadata["element_ids"]
        val_element_ids = ml_split.val.metadata["element_ids"]
        
        # Create tensor datasets with element IDs
        train_dataset = TensorDataset(
            torch.tensor(ml_split.train.X, dtype=torch.float32),
            torch.tensor(ml_split.train.y, dtype=torch.float32),
            torch.tensor(train_element_ids, dtype=torch.long)
        )
        
        val_dataset = TensorDataset(
            torch.tensor(ml_split.val.X, dtype=torch.float32),
            torch.tensor(ml_split.val.y, dtype=torch.float32),
            torch.tensor(val_element_ids, dtype=torch.long)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False
        )
        
        # Create trainer
        trainer = Trainer(
            max_epochs=self.cfg.max_epochs,
            accelerator="auto",
            devices=1,
            check_val_every_n_epoch=2,
            log_every_n_steps=1,
            callbacks=self.cfg.callbacks,
            default_root_dir=self.cfg.save_dir,
        )
        
        # Create lightning module
        lightning_module = ElementEmbeddingLightningModule(
            model=self.model.model,
            lr=self.cfg.initial_lr
        )
        
        # Train the model
        trainer.fit(
            lightning_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )
        
        # Update model with trained weights
        self.model.model = lightning_module.model
        
        return self
    
    def predict(self, X, element_ids=None):
        """
        Make predictions with the model.
        
        Args:
            X: Input features (n_samples, feature_dim)
            element_ids: Optional element IDs (n_samples, 1)
            
        Returns:
            Predictions (n_samples, output_dim)
        """
        from torch.utils.data import TensorDataset, DataLoader
        from lightning import Trainer
        
        # Use default element ID if not provided
        if element_ids is None:
            # Use the first element ID as default
            default_id = next(iter(self.element_to_id.values()))
            element_ids = np.full((len(X), 1), default_id)
        
        # Create tensor dataset
        dummy_y = torch.zeros((len(X), self.cfg.output_dim), dtype=torch.float32)
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            dummy_y,
            torch.tensor(element_ids, dtype=torch.long)
        )
        
        # Create data loader
        dataloader = DataLoader(dataset, batch_size=32)
        
        # Create trainer for prediction
        with torch.no_grad():
            self.model.model.eval()
            predictions = []
            
            for x, _, element_ids in dataloader:
                pred = self.model.model(x, element_ids)
                predictions.append(pred.numpy())
        
        return np.vstack(predictions)


def train_xes_model(
    ml_splits: MLSplits,
    input_dim: int = 64,
    output_dim: Optional[int] = None,
    hidden_dims: List[int] = [200, 200],
    max_epochs: int = 1000,
    initial_lr: float = 1e-3,
    batch_size: int = 32,
    directory: str = "xes_checkpoints",
    **kwargs
) -> XESBlockRegressor:
    """
    Train a model for XES prediction.
    
    Args:
        ml_splits: MLSplits containing train/val/test data
        input_dim: Input feature dimension
        output_dim: Output spectrum dimension (if None, determined from data)
        hidden_dims: Hidden layer dimensions
        max_epochs: Maximum training epochs
        initial_lr: Initial learning rate
        batch_size: Batch size for training
        directory: Directory to save model checkpoints
        **kwargs: Additional arguments for XESBlockRegressor
        
    Returns:
        Trained XESBlockRegressor model
    """
    # If output_dim not specified, infer from data
    if output_dim is None and ml_splits.train is not None:
        output_dim = ml_splits.train.y.shape[1]
    elif output_dim is None:
        raise ValueError("output_dim must be specified if ml_splits.train is None")
    
    # Create and train the model
    model = XESBlockRegressor(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        max_epochs=max_epochs,
        initial_lr=initial_lr,
        batch_size=batch_size,
        directory=directory,
        **kwargs
    )
    
    # Fit the model on the provided data
    model.fit(ml_splits)
    
    return model


def train_universal_xes_model(
    ml_splits: MLSplits,
    num_elements: int,
    element_to_id: Dict[str, int],
    input_dim: int = 64,
    output_dim: Optional[int] = None,
    embedding_dim: int = 16,
    hidden_dims: List[int] = [300, 300, 300],
    max_epochs: int = 1000,
    initial_lr: float = 1e-3,
    batch_size: int = 32,
    directory: str = "universal_xes_checkpoints",
    **kwargs
) -> UniversalXESBlockRegressor:
    """
    Train a universal model for XES prediction across multiple elements.
    
    Args:
        ml_splits: MLSplits containing train/val/test data with element IDs in metadata
        num_elements: Number of different elements to support
        element_to_id: Mapping from element symbols to integer IDs
        input_dim: Input feature dimension
        output_dim: Output spectrum dimension (if None, determined from data)
        embedding_dim: Dimension of element embeddings
        hidden_dims: Hidden layer dimensions
        max_epochs: Maximum training epochs
        initial_lr: Initial learning rate
        batch_size: Batch size for training
        directory: Directory to save model checkpoints
        **kwargs: Additional arguments
        
    Returns:
        Trained UniversalXESBlockRegressor model
    """
    # If output_dim not specified, infer from data
    if output_dim is None and ml_splits.train is not None:
        output_dim = ml_splits.train.y.shape[1]
    elif output_dim is None:
        raise ValueError("output_dim must be specified if ml_splits.train is None")
    
    # Create and train the model
    model = UniversalXESBlockRegressor(
        num_elements=num_elements,
        element_to_id=element_to_id,
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        max_epochs=max_epochs,
        initial_lr=initial_lr,
        batch_size=batch_size,
        directory=directory,
        **kwargs
    )
    
    # Fit the model on the provided data
    model.fit(ml_splits)
    
    return model 