import os
import glob
import numpy as np
from typing import List, Optional, Dict, Tuple
from pydantic import BaseModel, field_validator
from pymatgen.core import Molecule, Lattice, Structure

from omnixas.core.periodic_table import Element
from omnixas.data.ml_data import MLData, MLSplits


class XESSpectrum(BaseModel):
    """
    Container for X-ray Emission Spectroscopy (XES) data.
    
    Attributes:
        name: Unique identifier (often the file stem).
        element: Chemical element symbol, e.g. 'Co', 'Fe', etc.
        structure: The parsed geometry as a pymatgen Structure or Molecule.
        energies: A 1D array of energy values from the .txt file.
        intensities: The corresponding XES intensities (same length as energies).
    """
    name: str
    element: str
    structure: Molecule
    energies: np.ndarray
    intensities: np.ndarray
    
    @field_validator("energies", "intensities", mode="before")
    @classmethod
    def _to_numpy(cls, v):
        if isinstance(v, list):
            return np.array(v)
        return v

    class Config:
        arbitrary_types_allowed = True


def read_xyz_file(xyz_path: str) -> Molecule:
    """
    Parse a .xyz file into a pymatgen Molecule.
    
    Args:
        xyz_path: Path to the XYZ file
        
    Returns:
        Molecule: The parsed molecular structure
    """
    # Read directly as Molecule (which supports XYZ format)
    molecule = Molecule.from_file(xyz_path)
    return molecule


def read_xes_txt(xes_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a 2-column .txt file containing energies and intensities for XES.
    
    Args:
        xes_path: Path to the spectrum file
        
    Returns:
        Tuple of (energies, intensities) as numpy arrays
    """
    energies = []
    intensities = []
    with open(xes_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue  # skip blank or commented lines
            parts = line.split()
            if len(parts) < 2:
                continue  # skip malformed lines
            e_val = float(parts[0])
            i_val = float(parts[1])
            energies.append(e_val)
            intensities.append(i_val)
    return np.array(energies), np.array(intensities)


def find_site_index_for_element(structure: Structure, element_symbol: str) -> int:
    """
    Return the index of the site that matches element_symbol.
    Assumes exactly one site with that element is present.
    
    Args:
        structure: Pymatgen Structure or Molecule
        element_symbol: Chemical element symbol to find
        
    Returns:
        int: Index of the site with matching element
        
    Raises:
        ValueError: If no site with matching element is found
    """
    for i, site in enumerate(structure):
        if site.species_string == element_symbol:
            return i
    raise ValueError(f"No site of element {element_symbol} found in molecule.")


def gather_xes_data_for_element(element_dir: str, element_symbol: str) -> List[XESSpectrum]:
    """
    Gather XES data from subdirectories for a specific element.
    
    Args:
        element_dir: Path to element directory (e.g., "XES-3dtm/Co/")
        element_symbol: Element symbol (e.g., "Co")
        
    Returns:
        List of XESSpectrum objects
        
    Raises:
        FileNotFoundError: If required directories are missing
    """
    xyz_dir = os.path.join(element_dir, "xyz")
    xes_dir = os.path.join(element_dir, "xes")
    
    if not os.path.isdir(xyz_dir):
        raise FileNotFoundError(f"No folder named 'xyz' under {element_dir}")
    if not os.path.isdir(xes_dir):
        raise FileNotFoundError(f"No folder named 'xes' under {element_dir}")

    xes_data_list = []
    
    # Loop over all *.xyz files in xyz_dir
    xyz_files = sorted(glob.glob(os.path.join(xyz_dir, "*.xyz")))
    for xyz_file in xyz_files:
        base_name = os.path.splitext(os.path.basename(xyz_file))[0]
        
        # The matching .txt is in the xes folder with the same base name
        xes_file = os.path.join(xes_dir, base_name + ".txt")
        
        if not os.path.isfile(xes_file):
            # if the .txt is missing, skip
            continue
        
        # Parse geometry
        structure = read_xyz_file(xyz_file)
        
        # Parse XES data
        energies, intensities = read_xes_txt(xes_file)
        
        # Create the data entry
        data_entry = XESSpectrum(
            name=base_name,
            element=element_symbol,
            structure=structure,
            energies=energies,
            intensities=intensities
        )
        xes_data_list.append(data_entry)
        
    return xes_data_list 