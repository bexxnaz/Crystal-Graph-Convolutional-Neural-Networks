import torch
import logging
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from training.datasets.graph_builder import CrystalGraphBuilder

# Configure Logging
logger = logging.getLogger(__name__)

class CrystalGraphDataset(Dataset):
    """
    PyTorch Dataset for Crystal Graphs.
    
    Handlers conversions from raw material data (structures) to graph tensors.
    Supports in-memory caching to speed up training.
    """
    
    def __init__(self, 
                 data_list: List[Dict], 
                 builder: CrystalGraphBuilder, 
                 targets: List[str] = ["label_formation_energy", "label_band_gap", "label_density"],
                 cache_graphs: bool = True):
        """
        Args:
            data_list: List of dictionaries containing 'structure_obj' and properties.
            builder: Instance of CrystalGraphBuilder.
            targets: List of keys in data dict to use as regression targets.
            cache_graphs: If True, pre-compute all graphs during initialization.
        """
        self.data_list = data_list
        self.builder = builder
        self.targets = targets
        self.cached_graphs: Optional[List] = None
        
        if cache_graphs:
            self._cache_all_graphs()

    def _cache_all_graphs(self):
        """Pre-compute graphs for all structures."""
        logger.info("Caching crystal graphs...")
        self.cached_graphs = []
        
        success_count = 0
        for entry in tqdm(self.data_list, desc="Building Graphs"):
            struct = entry.get("structure_obj")
            if struct:
                graph = self.builder.get_graph(struct)
                self.cached_graphs.append(graph)
                success_count += 1
            else:
                # Fallback for missing structure (should be filtered upstream)
                logger.warning(f"Missing structure object for entry {entry.get('formula', 'unknown')}")
                # Append None to maintain index alignment, handle in getitem
                self.cached_graphs.append(None)
                
        logger.info(f"Cached {success_count}/{len(self.data_list)} graphs.")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        entry = self.data_list[idx]
        
        # 1. Get Graph
        if self.cached_graphs:
            graph = self.cached_graphs[idx]
        else:
            struct = entry.get("structure_obj")
            graph = self.builder.get_graph(struct) if struct else None
            
        if graph is None:
            # Return dummy zero tensors if generation failed
            # (In a real pipeline, we'd ensure data cleanliness beforehand)
            atom_fea = torch.zeros(1, 4)
            nbr_fea = torch.zeros(0, len(self.builder.gdf.filter))
            nbr_idx = torch.zeros(0, 2).long()
        else:
            atom_fea, nbr_fea, nbr_idx = graph

        # 2. Get Targets
        # Ensure we return a tensor
        target_vals = [entry.get(t) for t in self.targets]
        # Handle NoneTargets (if any)
        target_vals = [float(t) if t is not None else 0.0 for t in target_vals]
        target_tensor = torch.tensor(target_vals, dtype=torch.float32)

        return {
            "atom_fea": atom_fea,
            "nbr_fea": nbr_fea,
            "nbr_idx": nbr_idx,
            "target": target_tensor,
            "formula": entry.get("formula", "unknown"),
            "id": entry.get("id", str(idx))
        }

def collate_batch(dataset_list: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate a list of graph samples into a batch.
    
    Logic Explained:
    ---------------
    Standard batching (stacking) doesn't work for graphs because they have variable sizes
    (different numbers of atoms and edges).
    
    Instead, we create a "Disjoint Union" graph, which is conceptually one giant graph
    containing all the small graphs as isolated components.
    
    1. Atom Features: Simply concatenated [Sum(N_atoms), Feature_Dim].
    2. Edge Features: Simply concatenated [Sum(N_edges), Feature_Dim].
    3. Edge Indices: 
       - Since we concatenated atoms, the indices of the 2nd graph need to be shifted
         by the number of atoms in the 1st graph, and so on.
       - Formula: new_index = original_index + cumulative_atom_count
    4. Batch Vector:
       - We need to know which atom belongs to which original graph for pooling (readout).
       - We create a vector [0, 0, 0, 1, 1, 2, 2, 2, ...] mapping each atom to its batch index.
    
    Args:
        dataset_list: List of samples from CrystalGraphDataset
        
    Returns:
        Batch dictionary ready for CGCNN forward pass.
    """
    batch_atom_fea = []
    batch_nbr_fea = []
    batch_nbr_idx = []
    batch_target = []
    batch_ids = []
    batch_mapping = [] # Maps atom -> graph_index
    
    base_idx = 0
    for i, sample in enumerate(dataset_list):
        n_i = sample["atom_fea"].shape[0] # Number of atoms
        
        # 1. Features: Collect atomic features
        batch_atom_fea.append(sample["atom_fea"])
        
        # 2. Edges: Collect edge features
        batch_nbr_fea.append(sample["nbr_fea"])
        
        # 3. Indices: Shift indices by the current base atom count
        # This ensures edges point to the correct atoms in the large concatenated list
        batch_nbr_idx.append(sample["nbr_idx"] + base_idx)
        
        # 4. Targets: Collect regression targets
        batch_target.append(sample["target"])
        
        # 5. Metadata
        batch_ids.append(sample["id"])
        
        # 6. Batch mapping (segment_id): Create a vector of size n_i filled with the batch index i
        # Used for global pooling (scatter_add / scatter_mean)
        batch_mapping.append(torch.full((n_i,), i, dtype=torch.long))
        
        # Update base index for the next graph
        base_idx += n_i

    return {
        "atom_fea": torch.cat(batch_atom_fea, dim=0),
        "nbr_fea": torch.cat(batch_nbr_fea, dim=0),
        "nbr_idx": torch.cat(batch_nbr_idx, dim=0),
        "target": torch.stack(batch_target, dim=0),
        "batch": torch.cat(batch_mapping, dim=0),
        "ids": batch_ids
    }
