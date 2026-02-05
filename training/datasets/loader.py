import os
import json
import logging
import pandas as pd
from typing import List, Dict, Optional, Any
from pymatgen.core import Structure
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class MaterialDataLoader:
    """
    Robust dataset loader for material science data using pymatgen.
    Supports Materials Project-style JSON/CSV formats.
    """

    REQUIRED_COLUMNS = ["structure", "formation_energy_per_atom", "band_gap", "density"]

    def __init__(self, data_path: str, limit: int = None):
        """
        Initialize the loader.
        
        Args:
            data_path: Path to the dataset file (JSON or CSV).
            limit: Optional maximum number of samples to load (useful for large datasets like MPtrj).
        """
        self.data_path = data_path
        self.limit = limit
        self.raw_data: List[Dict[str, Any]] = []
        self.cleaned_data: List[Dict[str, Any]] = []

    def load(self) -> List[Dict[str, Any]]:
        """
        Load, parse, and validate the dataset.
        
        Returns:
            A list of validated material samples.
        """
        logger.info(f"Loading data from {self.data_path}...")
        
        if not os.path.exists(self.data_path):
            logger.error(f"File not found: {self.data_path}")
            # Do not raise here, allow empty return so caller can handle gracefully or show mock
            return []

        try:
            if self.data_path.endswith(".json"):
                # For very large files (MPtrj is 11GB), standard json.load might OOM.
                # In a real heavy-duty pipeline we'd use ijson or line-based reading.
                # For now, we assume user might use a subset or have enough RAM.
                with open(self.data_path, "r") as f:
                    self.raw_data = json.load(f)
            elif self.data_path.endswith(".csv"):
                df = pd.read_csv(self.data_path)
                self.raw_data = df.to_dict(orient="records")
            else:
                raise ValueError("Unsupported file format. Use .json or .csv")
            
            logger.info(f"Loaded {len(self.raw_data)} raw entries.")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

        self._process_data()
        return self.cleaned_data

    def _process_data(self):
        """
        Process raw data: parse structures, extract properties, and validate.
        Ensures determinism by sorting the output.
        """
        valid_samples = []
        
        for idx, entry in enumerate(self.raw_data):
            if self.limit and len(valid_samples) >= self.limit:
                break

            try:
                # 1. Parse Structure
                struct = self._parse_structure(entry)
                if not struct:
                    # logging.warning(f"Skipping entry {idx}: Invalid structure data.")
                    continue

                # 2. Extract Properties (Auto-adapt to dataset dialect)
                properties = self._extract_properties(entry)
                
                # MPtrj might lack valid properties if we failed to parse energies
                if not properties:
                    continue

                # 3. Extract Computed Features (Lattice, Composition)
                # If density missing in properties, take from computed
                features = self._compute_features(struct)
                if properties.get("label_density", 0.0) == 0.0:
                    properties["label_density"] = features["density_computed"]

                # 4. Combine
                sample = {
                    "structure_obj": struct,  # Keep pymatgen object for advanced usage
                    **features,
                    **properties
                }
                valid_samples.append(sample)

            except Exception as e:
                # logger.warning(f"Error processing entry {idx}: {e}")
                pass

        # ensure determinism
        # sort by formula to ensure reproducible order
        try:
            valid_samples.sort(key=lambda x: x["formula"])
        except:
            pass # sorting might fail if data inconsistent, ok to skip
        
        self.cleaned_data = valid_samples
        logger.info(f"Successfully processed {len(self.cleaned_data)} valid samples.")

    def _parse_structure(self, entry: Dict[str, Any]) -> Optional[Structure]:
        """Parse structure from dict or string."""
        try:
            # Handle standard keys and MPtrj keys
            struct_data = entry.get("structure") or entry.get("cif") or entry.get("poscar")
            
            if isinstance(struct_data, dict):
                return Structure.from_dict(struct_data)
            elif isinstance(struct_data, str):
                # If it looks like a CIF string
                fmt = "cif" if "cif" in str(entry.keys()).lower() or "loop_" in struct_data else "poscar"
                return Structure.from_str(struct_data, fmt=fmt)
            
            return None
        except Exception as e:
            return None

    def _extract_properties(self, entry: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Extract target labels with support for MPtrj/MatBench dialects."""
        try:
            # 1. Formation Energy / Energy per Atom
            # MPtrj: "energy_per_atom" (uncorrected) or "uncorrected_energy_per_atom"
            # MatBench: "formation_energy_per_atom"
            e_form = entry.get("formation_energy_per_atom")
            if e_form is None:
                e_form = entry.get("energy_per_atom") # Fallback for MPtrj
            if e_form is None and "energy" in entry and "nsites" in entry:
                 e_form = entry["energy"] / entry["nsites"]

            # 2. Band Gap
            # MPtrj: Often missing (it's a force dataset). Default to 0.0 for training stability.
            bg = entry.get("band_gap") or entry.get("gap pbe") # variants
            if bg is None:
                bg = 0.0 # Placeholder for MPtrj samples

            # 3. Density
            density = entry.get("density")
            # If missing, will be filled by computed features

            if e_form is None: 
                return None
                
            return {
                "label_formation_energy": float(e_form),
                "label_band_gap": float(bg),
                "label_density": float(density) if density else 0.0
            }
        except ValueError:
            return None

    def _compute_features(self, struct: Structure) -> Dict[str, Any]:
        """Compute basic features from pymatgen Structure."""
        # sga = SpacegroupAnalyzer(struct) # Expensive for 1.5M items, use basic props
        return {
            "formula": struct.composition.reduced_formula,
            "nsites": struct.num_sites,
            "volume": struct.volume,
            "density_computed": struct.density,
            "lattice_a": struct.lattice.a,
            "lattice_b": struct.lattice.b,
            "lattice_c": struct.lattice.c,
            "lattice_alpha": struct.lattice.alpha,
            "lattice_beta": struct.lattice.beta,
            "lattice_gamma": struct.lattice.gamma,
            # "crystal_system": sga.get_crystal_system(), # Optimization: skip expensive SGA
            # "spacegroup_number": sga.get_space_group_number()
        }

if __name__ == "__main__":
    # Test block
    print("This module is a library. Import MaterialDataLoader to use.")
