import numpy as np
import torch
from pymatgen.core import Structure
from typing import List, Dict, Tuple

class GaussianDistance:
    """
    Expands a distance value into a vector using a set of Gaussian basis functions.
    
    Formula: exp(-(d - mu)^2 / sigma^2)
    This is standard practice in CGCNN/MegNet to allow the network to learn 
    non-linear distance dependencies.
    """
    def __init__(self, dmin: float = 0.0, dmax: float = 8.0, step: float = 0.2, var: float = None):
        """
        Args:
            dmin: Minimum distance of interest (Angstrom).
            dmax: Maximum distance (cutoff radius).
            step: Step size between Gaussian centers.
            var: Variance (width) of the Gaussians. If None, set to step magnitude.
        """
        self.filter = np.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian expansion to an array of distances.
        
        Args:
            distances: Array of distances (N,).
            
        Returns:
            Expanded representation (N, n_basis_functions).
        """
        # (N, 1) - (1, n_basis) broadcasting
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 / self.var**2)

class CrystalGraphBuilder:
    """
    Converts pymatgen Structures into graph representations suitable for GNNs.
    
    Representation:
    - Nodes: Atoms (features: Z, Group, Period, Electronegativity).
    - Edges: Interatomic bonds (features: Gaussian-expanded distance).
    - Adjacency: Neighbor indices.
    """
    
    def __init__(self, radius: float = 8.0, max_neighbors: int = 12, dStep: float = 0.2):
        """
        Args:
            radius: Cutoff radius for finding neighbors (Angstrom).
            max_neighbors: Maximum neighbors to consider per atom (efficiency).
            dStep: Resolution of the Gaussian distance basis.
        """
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.gdf = GaussianDistance(dmin=0, dmax=radius, step=dStep)
        
    def get_node_features(self, structure: Structure) -> torch.Tensor:
        """
        Extract atom-level features.
        
        Why these features?
        - Z (Atomic Number): Fundamental identity of the atom.
        - Group: Determines valence electron configuration (chemistry).
        - Period: Correlates with atomic size and shielding.
        - Electronegativity (X): Driver of bond polarity and charge transfer.
        """
        features = []
        for site in structure:
            el = site.specie
            # Safe handling for electronegativity (noble gases often None/NaN)
            # Pymatgen might return NaN for noble gases with a warning
            raw_X = el.X
            if raw_X is None or np.isnan(raw_X):
                X = 0.0
            else:
                X = float(raw_X)
            
            features.append([
                float(el.Z),
                float(el.group),
                float(el.row),
                X
            ])
        return torch.tensor(features, dtype=torch.float32)

    def get_graph(self, structure: Structure) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build the graph components.
        
        Returns:
            atom_feas: Node features (n_atoms, n_node_features)
            nbr_feas: Edge features (n_edges, n_edge_features)
            nbr_idx: Adjacency indices (n_edges, 2) [src, dst]
        """
        # 1. Node Features
        atom_feas = self.get_node_features(structure)
        
        # 2. Get Neighbors
        # get_all_neighbors returns jagged list: [ [neighbors_for_atom_0], [neighbors_for_atom_1], ... ]
        # Each neighbor is (site, distance, index, image)
        all_neighbors = structure.get_all_neighbors(self.radius, include_index=True)
        
        all_nbrs = []
        for k, neighbors in enumerate(all_neighbors):
            # Sort by distance to keep the closest ones if we truncate
            neighbors = sorted(neighbors, key=lambda x: x[1])
            
            if len(neighbors) > self.max_neighbors:
                neighbors = neighbors[:self.max_neighbors]
            
            for nbr in neighbors:
                distance = nbr[1]
                nbr_index = nbr[2]
                
                # We store [src_index, nbr_index, distance] temporarily
                all_nbrs.append([k, nbr_index, distance])

        # Handle isolated atoms (rare in crystals, but possible in molecules/defects)
        if len(all_nbrs) == 0:
            # Self-loop as fallback or empty
            print(f"Warning: No neighbors found for structure {structure.formula}")
            nbr_feas = torch.zeros(0, len(self.gdf.filter))
            nbr_idx = torch.zeros(0, 2).long()
            return atom_feas, nbr_feas, nbr_idx

        all_nbrs = np.array(all_nbrs)
        
        # 3. Edge Features
        distances = all_nbrs[:, 2]
        nbr_feas_np = self.gdf.expand(distances)
        nbr_feas = torch.tensor(nbr_feas_np, dtype=torch.float32)
        
        # 4. Adjacency
        # PyTorch Geometric/GCN typically wants [2, n_edges], but [n_edges, 2] is fine too depending on usage.
        # Here we return [n_edges, 2] -> (Source, Destination)
        nbr_idx = torch.LongTensor(all_nbrs[:, :2])
        
        return atom_feas, nbr_feas, nbr_idx

if __name__ == "__main__":
    # Test Block
    print("Testing CrystalGraphBuilder...")
    
    # Create valid dummy structure (CsCl) using pymatgen
    # If pymatgen fails to download anything or we want to be offline safe:
    from pymatgen.core import Lattice
    
    # Simple Cubic Lattice
    coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
    lattice = Lattice.from_parameters(a=4.0, b=4.0, c=4.0, alpha=90, beta=90, gamma=90)
    struct = Structure(lattice, ["Cs", "Cl"], coords)
    
    builder = CrystalGraphBuilder(radius=5.0)
    
    nodes, edges, indices = builder.get_graph(struct)
    
    print("\nGraph properties for CsCl:")
    print(f"Num Atoms: {len(nodes)}")
    print(f"Node Feat Dim: {nodes.shape[1]} (Expected: 4)")
    print(f"Num Edges: {len(edges)}")
    print(f"Edge Feat Dim: {edges.shape[1]}")
    print(f"Sample Node (Cs): {nodes[0]}")
    
    assert nodes.shape[1] == 4
    assert len(edges) > 0
    print("\n✅ CrystalGraphBuilder verified.")
