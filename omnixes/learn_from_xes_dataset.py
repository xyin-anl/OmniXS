"""
This script demonstrates end-to-end processing of XES data:
1. Reading XES data from xyz/txt files
2. Featurizing with M3GNet
3. Splitting into train/val/test
4. Training a model
5. Evaluating performance
"""

import os
import json
import argparse
import numpy as np
import pickle
from typing import List, Dict, Any, Tuple
from pathlib import Path

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt

from omnixas.featurizer.m3gnet_featurizer import M3GNetSiteFeaturizer
from omnixas.data.ml_data import MLSplits, MLData

from omnixes import (
    XESSpectrum,
    gather_xes_data_for_element,
    process_element_data,
    ElementDataset,
    train_xes_model,
    train_universal_xes_model
)


def plot_spectra(
    energies,
    spectra_lists,
    labels=None,
    title=None,
    figsize=(10, 6),
    colors=None,
    alpha=0.8,
    linestyles=None,
    ylim=None,
    xlim=None,
):
    """
    Plot multiple spectra for comparison.
    
    Args:
        energies: List of energy arrays for each spectrum
        spectra_lists: List of lists of spectra (first dimension is spectrum type, second is sample)
        labels: Labels for each spectrum type (first dimension of spectra_lists)
        title: Title for the plot
        figsize: Figure size as (width, height)
        colors: List of colors for each spectrum type
        alpha: Transparency for the lines
        linestyles: List of linestyles for each spectrum type
        ylim: Y-axis limits as (min, max)
        xlim: X-axis limits as (min, max)
        
    Returns:
        fig, ax: The figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Default colors and linestyles
    if colors is None:
        colors = plt.cm.tab10.colors
    if linestyles is None:
        linestyles = ['-', '--', ':', '-.']
    
    # Ensure labels exist
    if labels is None:
        labels = [f"Spectrum {i+1}" for i in range(len(spectra_lists))]
    
    # Plot each set of spectra
    for i, spectra in enumerate(spectra_lists):
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        
        # Plot each individual spectrum in this set
        for j, spectrum in enumerate(spectra):
            if j == 0:  # Only add label for the first spectrum in each set
                ax.plot(energies[j], spectrum, color=color, linestyle=linestyle, 
                        alpha=alpha, label=labels[i])
            else:
                ax.plot(energies[j], spectrum, color=color, linestyle=linestyle, 
                        alpha=alpha)
    
    # Add title and labels
    if title:
        ax.set_title(title)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Intensity (a.u.)")
    
    # Set limits if provided
    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(alpha=0.3, linestyle='--')
    
    # Tight layout
    fig.tight_layout()
    
    return fig, ax


def process_all_elements(
    top_dir: str,
    elements: List[str],
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    seed: int = 42,
    pickle_dir: str = "pickled_datasets"
) -> Dict[str, ElementDataset]:
    """
    Process all elements' XES data into datasets ready for ML.
    
    Args:
        top_dir: Main data directory
        elements: List of element symbols to process
        train_fraction: Fraction for training
        val_fraction: Fraction for validation
        seed: Random seed for reproducibility
        pickle_dir: Directory to save/load pickled datasets
        
    Returns:
        Dictionary mapping element symbols to ElementDataset objects
    """
    # Create pickle directory if it doesn't exist
    os.makedirs(pickle_dir, exist_ok=True)
    
    # Initialize the site featurizer (shared for all elements)
    featurizer = M3GNetSiteFeaturizer()
    
    # Process each element
    results = {}
    for el in elements:
        # Define pickle file path for this element
        pickle_path = os.path.join(pickle_dir, f"{el}_dataset_{seed}_{train_fraction}_{val_fraction}.pkl")
        
        # Check if pickled dataset exists
        if os.path.exists(pickle_path):
            print(f"Loading pickled dataset for {el}...")
            try:
                with open(pickle_path, 'rb') as f:
                    element_dataset = pickle.load(f)
                    results[el] = element_dataset
                    print(f"  -> Loaded dataset for {el} from {pickle_path}")
                    print(f"     Train: {len(element_dataset.ml_splits.train.X)} samples")
                    print(f"     Val: {len(element_dataset.ml_splits.val.X)} samples")
                    print(f"     Test: {len(element_dataset.ml_splits.test.X)} samples")
                    print()
                    continue
            except Exception as e:
                print(f"  -> Error loading pickled dataset: {e}. Will reprocess.")
        
        element_path = os.path.join(top_dir, el)
        if not os.path.isdir(element_path):
            print(f"Warning: {element_path} not found. Skipping {el}.")
            continue
        
        print(f"Gathering XES data for {el}...")
        data_list = gather_xes_data_for_element(element_path, el)
        print(f"  -> Found {len(data_list)} matched XYZ/XES pairs.")
        
        if len(data_list) == 0:
            continue
        
        # Process the data into ML splits
        splits = process_element_data(
            data_list,
            featurizer,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            seed=seed
        )
        
        # Store in results
        element_dataset = ElementDataset(
            raw_data=data_list,
            ml_splits=splits,
            element=el
        )
        results[el] = element_dataset
        
        # Save the dataset as pickle
        try:
            with open(pickle_path, 'wb') as f:
                pickle.dump(element_dataset, f)
            print(f"  -> Saved dataset to {pickle_path}")
        except Exception as e:
            print(f"  -> Error saving pickled dataset: {e}")
        
        print(f"  -> Created splits for {el}:")
        print(f"     Train: {len(splits.train.X)} samples")
        print(f"     Val: {len(splits.val.X)} samples")
        print(f"     Test: {len(splits.test.X)} samples")
        print()
        
    return results


def create_universal_datasets(
    element_datasets: Dict[str, ElementDataset],
    element_to_id: Dict[str, int] = None
) -> Tuple[MLSplits, Dict[str, int]]:
    """
    Create a universal dataset by combining data from all elements.
    
    Args:
        element_datasets: Dictionary mapping element symbols to ElementDataset objects
        element_to_id: Optional mapping from element symbols to integer IDs
        
    Returns:
        Tuple of (universal_ml_splits, element_to_id_mapping)
    """
    # Create element-to-id mapping if not provided
    if element_to_id is None:
        elements = sorted(list(element_datasets.keys()))
        element_to_id = {element: i for i, element in enumerate(elements)}
    
    # Initialize combined data lists
    train_X_list, train_y_list = [], []
    val_X_list, val_y_list = [], []
    test_X_list, test_y_list = [], []
    
    train_element_ids, val_element_ids, test_element_ids = [], [], []
    
    # Combine data from all elements
    for element, dataset in element_datasets.items():
        element_id = element_to_id[element]
        
        # Add to train set
        if len(dataset.ml_splits.train.X) > 0:
            train_X_list.append(dataset.ml_splits.train.X)
            train_y_list.append(dataset.ml_splits.train.y)
            train_element_ids.extend([element_id] * len(dataset.ml_splits.train.X))
            
        # Add to validation set
        if len(dataset.ml_splits.val.X) > 0:
            val_X_list.append(dataset.ml_splits.val.X)
            val_y_list.append(dataset.ml_splits.val.y)
            val_element_ids.extend([element_id] * len(dataset.ml_splits.val.X))
            
        # Add to test set
        if len(dataset.ml_splits.test.X) > 0:
            test_X_list.append(dataset.ml_splits.test.X)
            test_y_list.append(dataset.ml_splits.test.y)
            test_element_ids.extend([element_id] * len(dataset.ml_splits.test.X))
    
    # Combine into arrays
    train_X = np.concatenate(train_X_list, axis=0) if train_X_list else np.array([])
    train_y = np.concatenate(train_y_list, axis=0) if train_y_list else np.array([])
    
    val_X = np.concatenate(val_X_list, axis=0) if val_X_list else np.array([])
    val_y = np.concatenate(val_y_list, axis=0) if val_y_list else np.array([])
    
    test_X = np.concatenate(test_X_list, axis=0) if test_X_list else np.array([])
    test_y = np.concatenate(test_y_list, axis=0) if test_y_list else np.array([])
    
    # Create element ID arrays
    train_element_ids = np.array(train_element_ids).reshape(-1, 1)
    val_element_ids = np.array(val_element_ids).reshape(-1, 1)
    test_element_ids = np.array(test_element_ids).reshape(-1, 1)
    
    # Create one-hot encoding for element IDs
    num_elements = len(element_to_id)
    
    # Create MLSplits for the universal model
    train_data = MLData(
        X=train_X, 
        y=train_y,
        metadata={"element_ids": train_element_ids, "element_to_id": element_to_id}
    )
    
    val_data = MLData(
        X=val_X, 
        y=val_y,
        metadata={"element_ids": val_element_ids, "element_to_id": element_to_id}
    )
    
    test_data = MLData(
        X=test_X, 
        y=test_y,
        metadata={"element_ids": test_element_ids, "element_to_id": element_to_id}
    )
    
    universal_splits = MLSplits(train=train_data, val=val_data, test=test_data)
    
    return universal_splits, element_to_id


def train_universal_xes_model(
    universal_splits: MLSplits,
    output_dir: str = "universal_xes_model",
    max_epochs: int = 1000
) -> Dict[str, Any]:
    """
    Train a universal XES model on combined data from all elements.
    
    Args:
        universal_splits: MLSplits containing combined data from all elements
        output_dir: Output directory for the model
        max_epochs: Maximum training epochs
        
    Returns:
        Dictionary with training results
    """
    print("Training universal XES model...")
    
    # Create model directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine output dimension from data
    output_dim = universal_splits.train.y.shape[1]
    
    # Get element mapping from metadata
    element_to_id = universal_splits.train.metadata["element_to_id"]
    num_elements = len(element_to_id)
    
    # Train the universal model with element embeddings
    from omnixes.xes_regressor import train_universal_xes_model as train_universal_model
    model = train_universal_model(
        ml_splits=universal_splits,
        num_elements=num_elements,
        element_to_id=element_to_id,
        output_dim=output_dim,
        max_epochs=max_epochs,
        directory=os.path.join(output_dir, "checkpoints"),
        # Use larger network for universal model
        hidden_dims=[300, 300, 300],
        # Use larger embedding for more element information
        embedding_dim=32
    )
    
    # Extract test element IDs for prediction
    test_element_ids = universal_splits.test.metadata["element_ids"]
    
    # Generate predictions
    test_predictions = model.predict(universal_splits.test.X, test_element_ids)
    
    # Calculate MSE on test set
    test_mse = np.mean((test_predictions - universal_splits.test.y) ** 2)
    
    # Store results
    results = {
        "test_mse": float(test_mse),
        "model_path": output_dir,
        "n_train": len(universal_splits.train.X),
        "n_test": len(universal_splits.test.X),
        "n_val": len(universal_splits.val.X),
        "output_dim": output_dim,
        "num_elements": num_elements,
        "elements": list(element_to_id.keys())
    }
    
    print(f"  -> Universal model test MSE: {test_mse:.4f}")
    print(f"  -> Model saved to {output_dir}")
    
    # Try to create plot of example predictions
    try:
        # Find example predictions for each element
        element_examples = {}
        element_results = {}
        
        # Get test indices for each element
        for element, element_id in element_to_id.items():
            # Find indices where test_element_ids matches this element
            element_indices = np.where(test_element_ids == element_id)[0]
            
            if len(element_indices) > 0:
                # Take up to 3 examples
                n_examples = min(3, len(element_indices))
                indices = element_indices[:n_examples]
                
                # Get predictions and ground truth
                element_preds = test_predictions[indices]
                element_truth = universal_splits.test.y[indices]
                
                # Calculate element-specific MSE
                element_mse = np.mean((element_preds - element_truth) ** 2)
                
                # Store for plotting
                element_examples[element] = (element_preds, element_truth)
                element_results[element] = element_mse
        
        # Save element-specific results
        element_results_path = os.path.join(output_dir, "element_results.json")
        with open(element_results_path, "w") as f:
            json.dump(element_results, f, indent=2)
        
        # Create plots for each element
        for element, (preds, truth) in element_examples.items():
            # TODO: Need to get energies for plotting - assuming common energy scale
            
            plot_path = os.path.join(output_dir, f"{element}_predictions.png")
            try:
                # Construct dummy energies if needed
                dummy_energies = np.arange(preds.shape[1])
                
                # Plot predictions vs ground truth
                fig, axes = plot_spectra(
                    energies=[dummy_energies] * len(preds),
                    spectra_lists=[preds, truth],
                    labels=["Universal Model Prediction", "Ground Truth"],
                    title=f"Universal XES Predictions for {element}",
                    figsize=(10, 6)
                )
                fig.savefig(plot_path)
                print(f"  -> Plot saved to {plot_path}")
            except Exception as e:
                print(f"  -> Could not create plot for {element}: {e}")
    except Exception as e:
        print(f"  -> Could not create element-specific plots: {e}")
    
    # Save summary
    summary_path = os.path.join(output_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return model, results


def finetune_element_models(
    universal_model,
    element_datasets: Dict[str, ElementDataset],
    output_dir: str = "finetuned_xes_models",
    max_epochs: int = 200
) -> Dict[str, Any]:
    """
    Finetune the universal model for each element.
    
    Args:
        universal_model: Trained universal XES model
        element_datasets: Dictionary of element datasets
        output_dir: Output directory for models
        max_epochs: Maximum finetuning epochs
        
    Returns:
        Dictionary with finetuning results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # Get element mapping from universal model
    element_to_id = universal_model.element_to_id
    
    for element, dataset in element_datasets.items():
        print(f"Finetuning model for {element}...")
        
        # Skip if no data
        if len(dataset.ml_splits.train.X) == 0:
            print(f"  -> No training data for {element}. Skipping.")
            continue
        
        # Skip if element not in universal model
        if element not in element_to_id:
            print(f"  -> Element {element} not included in universal model. Skipping.")
            continue
            
        # Create model directory
        model_dir = os.path.join(output_dir, element)
        os.makedirs(model_dir, exist_ok=True)
        
        # Determine output dimension from data
        output_dim = dataset.ml_splits.train.y.shape[1]
        
        # Get the element ID for this element
        element_id = element_to_id[element]
        
        # Add element ID to dataset metadata for training
        train_element_ids = np.full((len(dataset.ml_splits.train.X), 1), element_id)
        val_element_ids = np.full((len(dataset.ml_splits.val.X), 1), element_id)
        test_element_ids = np.full((len(dataset.ml_splits.test.X), 1), element_id)
        
        # Create new datasets with element IDs
        train_data = MLData(
            X=dataset.ml_splits.train.X, 
            y=dataset.ml_splits.train.y,
            metadata={"element_ids": train_element_ids, "element_to_id": {element: element_id}}
        )
        
        val_data = MLData(
            X=dataset.ml_splits.val.X, 
            y=dataset.ml_splits.val.y,
            metadata={"element_ids": val_element_ids, "element_to_id": {element: element_id}}
        )
        
        test_data = MLData(
            X=dataset.ml_splits.test.X, 
            y=dataset.ml_splits.test.y,
            metadata={"element_ids": test_element_ids, "element_to_id": {element: element_id}}
        )
        
        # Create ML splits
        element_splits = MLSplits(train=train_data, val=val_data, test=test_data)
        
        # Clone the universal model
        import copy
        from omnixes.xes_regressor import UniversalXESBlockRegressor
        
        # Create a new model with the same architecture but only this element for finetuning
        finetuned_model = UniversalXESBlockRegressor(
            num_elements=universal_model.num_elements,  # Keep original number
            element_to_id=element_to_id,  # Keep original mapping
            input_dim=64,  # M3GNet features
            embedding_dim=universal_model.embedding_dim,
            output_dim=output_dim,
            hidden_dims=[300, 300, 300],  # Same as universal model
            max_epochs=max_epochs,
            initial_lr=1e-4,  # Lower learning rate for finetuning
            batch_size=32,
            directory=os.path.join(model_dir, "checkpoints")
        )
        
        # Copy the model weights
        finetuned_model.model.model.load_state_dict(
            copy.deepcopy(universal_model.model.model.state_dict())
        )
        
        # Finetune on element-specific data
        finetuned_model.fit(element_splits)
        
        # Generate predictions
        test_predictions = finetuned_model.predict(element_splits.test.X, test_element_ids)
        
        # Calculate MSE on test set
        test_mse = np.mean((test_predictions - element_splits.test.y) ** 2)
        
        # Store results
        results[element] = {
            "test_mse": float(test_mse),
            "model_path": model_dir,
            "n_train": len(element_splits.train.X),
            "n_test": len(element_splits.test.X),
            "n_val": len(element_splits.val.X),
            "output_dim": output_dim
        }
        
        print(f"  -> Test MSE: {test_mse:.4f}")
        print(f"  -> Model saved to {model_dir}")
        
        # Plot example predictions
        plot_path = os.path.join(model_dir, "test_predictions.png")
        try:
            # Get some test examples
            n_samples = min(5, len(test_predictions))
            
            # Get the energy values from the first raw spectrum
            example_energies = dataset.raw_data[0].energies
            
            # Plot the predictions vs ground truth
            fig, axes = plot_spectra(
                energies=[example_energies] * n_samples,
                spectra_lists=[
                    test_predictions[:n_samples],
                    element_splits.test.y[:n_samples]
                ],
                labels=["Prediction (Finetuned)", "Ground Truth"],
                title=f"XES Predictions for {element} (Finetuned)",
                figsize=(10, 6)
            )
            fig.savefig(plot_path)
            print(f"  -> Plot saved to {plot_path}")
        except Exception as e:
            print(f"  -> Could not create plot: {e}")
        
        print()
    
    # Save summary
    summary_path = os.path.join(output_dir, "finetuning_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return results
        

def train_models(
    element_datasets: Dict[str, ElementDataset],
    output_dir: str = "xes_models",
    max_epochs: int = 1000
) -> Dict[str, Any]:
    """
    Train XES prediction models for each element.
    
    Args:
        element_datasets: Dictionary of element datasets
        output_dir: Output directory for models
        max_epochs: Maximum training epochs
        
    Returns:
        Dictionary with training results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    for element, dataset in element_datasets.items():
        print(f"Training model for {element}...")
        
        # Skip if no data
        if len(dataset.ml_splits.train.X) == 0:
            print(f"  -> No training data for {element}. Skipping.")
            continue
        
        # Create model directory
        model_dir = os.path.join(output_dir, element)
        os.makedirs(model_dir, exist_ok=True)
        
        # Determine output dimension from data
        output_dim = dataset.ml_splits.train.y.shape[1]
        
        # Train the model
        model = train_xes_model(
            dataset.ml_splits,
            output_dim=output_dim,
            max_epochs=max_epochs,
            directory=os.path.join(model_dir, "checkpoints")
        )
        
        # Generate predictions
        test_predictions = model.predict(dataset.ml_splits.test.X)
        
        # Calculate MSE on test set
        test_mse = np.mean((test_predictions - dataset.ml_splits.test.y) ** 2)
        
        # Store results
        results[element] = {
            "test_mse": float(test_mse),
            "model_path": model_dir,
            "n_train": len(dataset.ml_splits.train.X),
            "n_test": len(dataset.ml_splits.test.X),
            "n_val": len(dataset.ml_splits.val.X),
            "output_dim": output_dim
        }
        
        print(f"  -> Test MSE: {test_mse:.4f}")
        print(f"  -> Model saved to {model_dir}")
        
        # Plot example predictions
        plot_path = os.path.join(model_dir, "test_predictions.png")
        try:
            # Get some test examples
            n_samples = min(5, len(test_predictions))
            
            # Get the energy values from the first raw spectrum
            example_energies = dataset.raw_data[0].energies
            
            # Plot the predictions vs ground truth
            fig, axes = plot_spectra(
                energies=[example_energies] * n_samples,
                spectra_lists=[
                    test_predictions[:n_samples],
                    dataset.ml_splits.test.y[:n_samples]
                ],
                labels=["Prediction", "Ground Truth"],
                title=f"XES Predictions for {element}",
                figsize=(10, 6)
            )
            fig.savefig(plot_path)
            print(f"  -> Plot saved to {plot_path}")
        except Exception as e:
            print(f"  -> Could not create plot: {e}")
        
        print()
    
    # Save summary
    summary_path = os.path.join(output_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return results
        

def main():
    """Main function to process data and train models."""
    parser = argparse.ArgumentParser(description="Process XES data and/or train models.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--data_dir", type=str, default="XES-3dtm",
                        help="Path to the data directory (default: XES-3dtm)")
    parser.add_argument("--elements", type=str, nargs="+",
                        default=["Co", "Cr", "Cu", "Fe", "Mn", "Ni", "Ti", "V", "Zn"],
                        help="Elements to process")
    parser.add_argument("--output_dir", type=str, default="xes_models",
                        help="Output directory for models")
    parser.add_argument("--train_fraction", type=float, default=0.8,
                        help="Fraction of data for training")
    parser.add_argument("--val_fraction", type=float, default=0.1,
                        help="Fraction of data for validation")
    
    parser.add_argument("--max_epochs", type=int, default=1000,
                        help="Maximum training epochs")
    
    parser.add_argument("--pickle_dir", type=str, default="pickled_datasets",
                        help="Directory to save/load pickled datasets")
    parser.add_argument("--force_reprocess", action="store_true",
                        help="Force reprocessing even if pickled datasets exist")
    
    # Universal model options
    parser.add_argument("--train_universal", action="store_true",
                        help="Train a universal model on all elements")
    parser.add_argument("--finetune", action="store_true",
                        help="Finetune the universal model for each element")
    parser.add_argument("--universal_output_dir", type=str, default="universal_xes_model",
                        help="Output directory for universal model")
    parser.add_argument("--finetuned_output_dir", type=str, default="finetuned_xes_models",
                        help="Output directory for finetuned models")
    
    args = parser.parse_args()
    
    pickle_dir = args.pickle_dir
    if args.force_reprocess:
        # If forcing reprocessing, rename the pickle directory to avoid using cached files
        import time
        timestamp = int(time.time())
        pickle_dir = f"{args.pickle_dir}_{timestamp}"
    
    # Process all elements
    element_datasets = process_all_elements(
        args.data_dir,
        args.elements,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
        pickle_dir=pickle_dir
    )
    
    # Train individual models if not skipped
    if not (args.train_universal and not args.finetune):
        element_results = train_models(
            element_datasets,
            output_dir=args.output_dir,
            max_epochs=args.max_epochs
        )
    
    # Train universal model if requested
    universal_model = None
    if args.train_universal:
        # Create universal dataset
        universal_splits, element_to_id = create_universal_datasets(element_datasets)
        
        # Train universal model
        universal_model, universal_results = train_universal_xes_model(
            universal_splits,
            output_dir=args.universal_output_dir,
            max_epochs=args.max_epochs
        )
        
        # Finetune if requested
        if args.finetune:
            finetune_results = finetune_element_models(
                universal_model,
                element_datasets,
                output_dir=args.finetuned_output_dir,
                max_epochs=max(200, args.max_epochs // 5)  # Shorter finetuning
            )
    
    # Print summary of individual element models
    if not (args.train_universal and not args.finetune):
        print("\nIndividual Model Training Summary:")
        for element, result in element_results.items():
            print(f"{element}: Test MSE = {result['test_mse']:.4f}, "
                  f"Train samples = {result['n_train']}")
    
    # Print summary of finetuned models if applicable
    if args.train_universal and args.finetune:
        print("\nFinetuned Model Summary:")
        for element, result in finetune_results.items():
            print(f"{element}: Test MSE = {result['test_mse']:.4f}, "
                  f"Train samples = {result['n_train']}")
    

if __name__ == "__main__":
    main() 