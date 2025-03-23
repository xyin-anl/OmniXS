# OmniXES

This module provides functionality to process X-ray Emission Spectroscopy (XES) data. It handles the complete pipeline from data loading through to model training.

## Overview

The module includes functionality to:
1. Read `.xyz` files containing atomic structures and `.txt` files containing XES spectra
2. Match corresponding files and parse them into `XESSpectrum` objects
3. Featurize structures using the M3GNet featurizer from OmniXAS
4. Split data into train/validation/test sets
5. Train machine learning models to predict XES spectra
6. Evaluate model performance

## Directory Structure

The expected directory structure for your data is:

```
XES_data/
└── <Element>/
    ├── xyz/
    │    ├── sample1.xyz
    │    ├── sample2.xyz
    │    └── ...
    ├── xyz-ho/      # (hold-out xyz)
    ├── xes/
    │    ├── sample1.txt
    │    ├── sample2.txt
    │    └── ...
    └── xes-ho/      # (hold-out xes)
```

Files named `sample1.xyz` will match with `sample1.txt` in the `xes` subfolder.

## Usage

### Installation

Follow the same installation instructions for OmniXAS.

### Command Line Usage

You can also use the included command-line script:

```bash
python omnixes/learn_from_xes_dataset.py --data_dir XES_data --elements Co Fe Ni
```

Options:
- `--data_dir`: Path to the data directory (default: "XES-3dtm")
- `--elements`: List of elements to process (default: all common transition metals)
- `--output_dir`: Directory to save models (default: "xes_models")
- `--train_fraction`: Fraction of data for training (default: 0.8)
- `--val_fraction`: Fraction of data for validation (default: 0.1)
- `--seed`: Random seed (default: 42)
- `--max_epochs`: Maximum training epochs (default: 1000)

## Classes and Functions

### Main Classes

- `XESSpectrum`: Container for XES data including structure and spectrum
- `ElementDataset`: Container for processed element data and ML splits
- `XESBlockRegressor`: Neural network model for XES prediction

### Key Functions

- `gather_xes_data_for_element()`: Read and parse XES data from files
- `featurize_xes_data()`: Convert XES data to ML-ready features
- `process_element_data()`: Process and split data for one element
- `train_xes_model()`: Train a model on XES data

## Integration with OmniXAS

This module reuses key components from OmniXAS:

1. M3GNet featurizer for structural representation
2. MLData/MLSplits for data organization
3. XASBlock neural network architecture (repurposed for XES)
4. Lightning-based training infrastructure

The training pipeline mimics the approach used in OmniXAS, but with adjustments for XES data, particularly the output dimension which corresponds to the spectrum length.