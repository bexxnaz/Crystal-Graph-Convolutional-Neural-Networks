import numpy as np
import pandas as pd
from typing import List, Dict, Union
from pymatgen.core import Composition, Element

class ChemicalFeaturizer:
    """
    Composition-based feature extractor for inorganic materials.
    
    This 'Magpie-style' featurizer aggregates element-level properties
    to create a fixed-length vector representation of a material.
    
    Why it works:
    Material properties are often dominated by the average or extreme values 
    of their constituent elements. For example:
    - Average electronegativity correlates with bond ionicity (Band Gap).
    - Mean atomic weight correlates with density.
    - Max valence electron count can constrain stability.
    """

    # Properties to extract for each element
    ELEMENT_PROPERTIES = [
        "Z",            # Atomic number
        "atomic_mass",  # Atomic mass
        "X",            # Electronegativity (Pauling)
        "atomic_radius",# Atomic radius
        "n_valence"     # Number of valence electrons (approximate)
    ]
    
    # Aggregation functions to apply
    STATS = ["mean", "max", "min", "std"]

    def __init__(self):
        """Initialize the featurizer."""
        self.feature_labels = self._generate_labels()

    def featurize_formula(self, formula: str) -> np.ndarray:
        """
        Convert a chemical formula string into a feature vector.
        
        Args:
            formula: e.g., "Li2FeO3"
            
        Returns:
            A 1D numpy array of features.
        """
        try:
            comp = Composition(formula)
            return self._featurize_composition(comp)
        except Exception as e:
            # Return zero vector on failure (or handle as needed)
            print(f"Featurization error for {formula}: {e}")
            return np.zeros(len(self.feature_labels))

    def _featurize_composition(self, comp: Composition) -> np.ndarray:
        """Internal method to compute statistics from a Composition object."""
        # Get element objects and their fractions
        elements = comp.elements
        fractions = np.array([comp.get_atomic_fraction(el) for el in elements])
        
        # Collect property values for each element
        prop_matrix = []
        for el in elements:
            props = self._get_element_properties(el)
            prop_matrix.append(props)
        
        prop_matrix = np.array(prop_matrix) # Shape: (n_elements, n_properties)
        
        # Compute statistics
        # Mean is weighted by atomic fraction
        mean_vals = np.average(prop_matrix, axis=0, weights=fractions)
        max_vals = np.max(prop_matrix, axis=0)
        min_vals = np.min(prop_matrix, axis=0)
        
        # Weighted standard deviation
        variance = np.average((prop_matrix - mean_vals)**2, axis=0, weights=fractions)
        std_vals = np.sqrt(variance)
        
        # Concatenate all stats
        feature_vector = np.concatenate([mean_vals, max_vals, min_vals, std_vals])
        return feature_vector

    def _get_element_properties(self, el: Element) -> List[float]:
        """Extract atomic properties from pymatgen Element."""
        # Valence approximation (p-block valence can vary, taking simple view)
        # Using a safer fallback for missing data
        X = el.X if el.X else 0.0
        radius = el.atomic_radius if el.atomic_radius else 0.0
        
        # Pymatgen doesn't expose a single 'n_valence' simply for all elements in the periodic table 
        # that fits ML needs perfectly, but group number is a decent proxy for s/p blocks.
        # Here we use a safe simple lookup or standard property.
        group = el.group
        
        return [
            float(el.Z),
            float(el.atomic_mass),
            float(X),
            float(radius),
            float(group) # Using group number as a proxy for valence behavior context
        ]

    def _generate_labels(self) -> List[str]:
        """Generate human-readable labels for the features."""
        labels = []
        prop_names = ["AtomicNumber", "AtomicMass", "Electronegativity", "AtomicRadius", "GroupNumber"]
        
        for stat in self.STATS:
            for prop in prop_names:
                labels.append(f"{stat}_{prop}")
        
        return labels

if __name__ == "__main__":
    # verification block
    featurizer = ChemicalFeaturizer()
    formula = "Li2FeO3"
    vector = featurizer.featurize_formula(formula)
    
    print(f"Featurizing {formula}...")
    print(f"Dimension: {len(vector)}")
    print("Feature Vector Sample:")
    df = pd.DataFrame([vector], columns=featurizer.feature_labels)
    print(df.iloc[0].head(10))
