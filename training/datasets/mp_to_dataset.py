"""
Data collection from Materials Project API.
Downloads crystal structures with their properties.
"""

import os
import json
from typing import List, Dict, Optional, Tuple
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from mp_api.client import MPRester
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

TRANSITION_METALS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
    "Y", "Zr", "Nb", "Mo", "Hf", "Ta", "W"
}

A_ELEMENTS = {
    "Al", "Si", "Ga", "Ge", "Sn", "In"
}

X_ELEMENTS = {"C", "N"}


class MaterialsProjectCollector:
    """
    Collect crystal structures and properties from Materials Project.

    Features:
    - Download structures with target properties
    - Filter by composition, stability, properties
    - Save to structured format
    """

    def __init__(self, api_key: Optional[str] = None, save_dir: str = "data/materials_project"):
        """
        Initialize collector.

        Args:
            api_key: Materials Project API key (or set MP_API_KEY env variable)
            save_dir: Directory to save downloaded data
        """
        self.api_key = api_key or os.environ.get("MP_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Materials Project API key required. "
                "Get one at https://materialsproject.org/api"
            )

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.mpr = MPRester(self.api_key)
    
    def filter_max_phases(self, data: List[Dict]) -> List[Dict]:
        """
        Filter MAX phases from a collected Materials Project dataset.

        Args:
            data: List of material entries (dicts)

        Returns:
            List containing only MAX-phase materials
        """
        return [d for d in data if d.get("is_max_phase", False)]

    def collect_structures(
        self,
        num_structures: int = 20000,
        elements: Optional[List[str]] = None,
        exclude_elements: Optional[List[str]] = None,
        max_sites: int = 50,
        e_above_hull_max: float = 0.1,
        properties: List[str] = None,
        save_name: str = "materials_data"
    ) -> pd.DataFrame:
        """
        Collect crystal structures with properties.

        Args:
            num_structures: Maximum number of structures to download
            elements: List of elements to include (e.g., ['Li', 'Fe', 'O'])
            exclude_elements: Elements to exclude
            max_sites: Maximum number of atoms per structure
            e_above_hull_max: Maximum energy above hull (eV/atom) for stability
            properties: Properties to download
            save_name: Name for saved files

        Returns:
            DataFrame with structures and properties
        """
        if properties is None:
            properties = [
                "material_id",
                "formula_pretty",
                "structure",
                "formation_energy_per_atom",
                "energy_above_hull",
                "band_gap",
                "density",
                "volume",
                "nsites",
                "elements",
                "nelements",
            ]

        print(f"Collecting structures from Materials Project...")
        print(f"Filters: elements={elements}, max_sites={max_sites}, "
              f"e_above_hull<={e_above_hull_max}")

        # Build query
        query_params = {
            "num_sites": (1, max_sites),
            "energy_above_hull": (0, e_above_hull_max),
        }

        if elements:
            query_params["elements"] = elements
        if exclude_elements:
            query_params["exclude_elements"] = exclude_elements

        # Query Materials Project
        try:
            docs = self.mpr.summary.search(
                **query_params,
                fields=properties,
                num_chunks=80,   
                chunk_size=1000
            )
        except Exception as e:
            raise RuntimeError(f"Materials Project query failed: {e}")

        print(f"Found {len(docs)} structures")

        # # Limit to requested number
        # if len(docs) > num_structures:
        #     docs = docs[:num_structures]
        #     print(f"Limited to {num_structures} structures")

        # Process structures
        data = []
        for doc in tqdm(docs, desc="Processing structures"):
            try:
                entry = self._process_document(doc)
                if entry:
                    data.append(entry)
            except Exception as e:
                print(f"Error processing {doc.material_id}: {e}")
                continue

        # Save data
        self._save_dataset(data, save_name)
        return data
    def is_max_phase(self, elements: List[str]) -> bool:
        elements = set(elements)

        if len(elements) != 3:
            return False

        has_M = len(elements & TRANSITION_METALS) >= 1
        has_A = len(elements & A_ELEMENTS) >= 1
        has_X = len(elements & X_ELEMENTS) >= 1

        return has_M and has_A and has_X

    def _process_document(self, doc) -> Optional[Dict]:
        try:
            if doc.formation_energy_per_atom is None:
                return None

            structure = doc.structure

            try:
                sga = SpacegroupAnalyzer(structure)
                space_group = sga.get_space_group_number()
                crystal_system = sga.get_crystal_system()
            except Exception:
                space_group = None
                crystal_system = None

            e_above_hull = doc.energy_above_hull if doc.energy_above_hull is not None else 0.0

            entry = {
                "material_id": doc.material_id,
                "formula": doc.formula_pretty,
                
                "structure": structure.as_dict(),

                "num_sites": structure.num_sites,
                "volume": float(structure.volume),
                "density": float(doc.density) if doc.density is not None else float(structure.density),

                "formation_energy_per_atom": float(doc.formation_energy_per_atom),
                "band_gap": float(doc.band_gap) if doc.band_gap is not None else 0.0,
                "energy_above_hull": float(e_above_hull),

                # labels
                "is_stable": bool(e_above_hull < 0.01),

                # composition info
                "elements": [str(el) for el in structure.composition.elements],
                "num_elements": len(structure.composition.elements),

                # symmetry info
                "space_group": space_group,
                "crystal_system": crystal_system,
                "is_max_phase": self.is_max_phase(
                    [str(el) for el in structure.composition.elements]),

            }

            return entry

        except Exception as e:
            print(f"Error processing structure: {e}")
            return None


    def _alternative_query(self, query_params, properties):
        """Alternative query method if main one fails."""
        # Simplified query
        docs = []
        try:
            # Use basic search
            results = self.mpr.summary.search(
                num_sites=(1, query_params.get("num_sites", (1, 50))[1]),
                fields=["material_id", "formula_pretty", "structure",
                       "formation_energy_per_atom", "band_gap", "density"]
            )
            docs = results
        except Exception as e:
            print(f"Alternative query also failed: {e}")

        return docs

    def _save_dataset(self, data: List[Dict], save_name: str):
        """Save dataset as a single JSON file (GNN-ready)."""
        
        json_path = self.save_dir / f"{save_name}.json"

        with open(json_path, "w") as f:
            json.dump(data, f)

        print(f"\n✅ Saved dataset to {json_path}")
        print(f"Total samples: {len(data)}")

    def load_dataset(self, save_name: str = "materials_data") -> pd.DataFrame:
        """Load previously saved dataset."""
        pkl_path = self.save_dir / f"{save_name}_full.pkl"
        if pkl_path.exists():
            return pd.read_pickle(pkl_path)
        else:
            raise FileNotFoundError(f"Dataset not found: {pkl_path}")

    def collect_by_material_class(
        self,
        material_class: str,
        num_structures: int = 5000
    ) -> pd.DataFrame:
        """
        Collect structures for specific material classes.

        Args:
            material_class: 'perovskites', 'spinels', 'garnets', 'alloys', etc.
            num_structures: Number of structures to collect

        Returns:
            DataFrame with structures
        """
        class_filters = {
            'perovskites': {
                'exclude_elements': ['Hg', 'Po', 'Rn', 'Ac', 'Pa', 'Np', 'Pu'],
            },
            'binary_oxides': {
                'elements': ['O'],
                'exclude_elements': ['Hg', 'Po', 'Rn'],
            },
            'battery_materials': {
                'elements': ['Li'],
                'exclude_elements': ['Hg', 'Po', 'Rn'],
            },
            'alloys': {
                'exclude_elements': ['O', 'N', 'F', 'Cl', 'Br', 'I', 'S', 'Se', 'Te'],
            },
        }

        if material_class in class_filters:
            filters = class_filters[material_class]
            return self.collect_structures(
                num_structures=num_structures,
                **filters,
                save_name=f"materials_{material_class}"
            )
        else:
            raise ValueError(f"Unknown material class: {material_class}")


def main():
    """Example usage."""
    # Initialize collector (requires MP_API_KEY environment variable)
    try:
        collector = MaterialsProjectCollector()

        # Collect diverse structures
        print("Collecting diverse materials...")
        df = collector.collect_structures(
          num_structures=20000,
          max_sites=50,
          e_above_hull_max=0.2,
          save_name="all_data"
        )


        # Filter MAX phases
        max_data = collector.filter_max_phases(df)

        print(f"MAX phases found: {len(max_data)}")

        # Save MAX-only dataset
        collector._save_dataset(max_data, "mp_max")


        print(f"\nCollected {len(df)} structures")

        # # Collect perovskites
        # print("\n" + "="*50)
        # print("Collecting perovskites...")
        # df_perovskites = collector.collect_by_material_class(
        #     'perovskites',
        #     num_structures=2000
        # )

        # print(f"Collected {len(df_perovskites)} perovskite structures")

    except Exception as e:
        print(f"Error: {e}")
        print("\nPlease set your Materials Project API key:")
        print("export MP_API_KEY='your_api_key_here'")
        print("\nGet your API key at: https://materialsproject.org/api")


if __name__ == "__main__":
    main()



