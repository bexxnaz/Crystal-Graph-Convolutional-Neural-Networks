import torch
import torch.nn as nn
import torch.nn.functional as F

class CGCNNConv(nn.Module):
    """
    Crystal Graph Convolutional Neural Network (CGCNN) Layer.
    
    Paper: Xie, T., & Grossman, J. C. (2018). Crystal Graph Convolutional Neural Networks 
           for an Accurate and Interpretable Prediction of Material Properties. 
           Physical Review Letters.
           
    Physics Intuition:
    ------------------
    Atoms (nodes) interact with their neighbors (edges) to update their state vector (features).
    The interaction strength is determined by the local chemical environment (neighbor type)
    and geometry (bond length).
    
    The visualization of this process is:
    v_i_new = v_i + Sum_j [ sigmoid(z_ij * W_f + b_f) * g(z_ij * W_s + b_s) ]
    
    Where:
    - v_i: central atom features
    - z_ij: concatenated vector [vi, vj, u_ij] (central, neighbor, bond)
    - sigmoid(...) term: "Soft Filter" or Attention. Determines HOW MUCH the neighbor influences.
                         Analogous to bond strength or screening.
    - g(...) term: "Core Signal". The actual chemical information being transmitted.
                   (Usually a softplus or tanh/linear function).
    """

    def __init__(self, node_dim: int, edge_dim: int):
        """
        Args:
            node_dim: Dimension of atom features.
            edge_dim: Dimension of bond features (Gaussian distance).
        """
        super(CGCNNConv, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        
        # The input to the linear layers is the concatenation of:
        # Atom_i (node_dim) + Atom_j (node_dim) + Bond_ij (edge_dim)
        full_dim = 2 * node_dim + edge_dim
        
        # Soft Gating Filter (Sigmoid)
        self.fc_filter = nn.Linear(full_dim, node_dim)
        self.bn_filter = nn.BatchNorm1d(node_dim)
        
        # Core Signal (Softplus or Tanh - CGCNN uses Softplus typically, but Tanh is common in standard GNNs)
        # We'll use Softplus as per original implementation preference for positive features, 
        # or Linear/Tanh. Let's stick to Softplus for stability in materials.
        self.fc_core = nn.Linear(full_dim, node_dim)
        self.bn_core = nn.BatchNorm1d(node_dim)

    def forward(self, atom_features: torch.Tensor, edge_features: torch.Tensor, edge_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            atom_features: (N_atoms, node_dim)
            edge_features: (N_edges, edge_dim)
            edge_indices: (N_edges, 2) -> [source, target]
            
        Returns:
            updated_atom_features: (N_atoms, node_dim)
        """
        # 1. Gather Features
        # edge_indices[:, 0] are source atoms (j)
        # edge_indices[:, 1] are target atoms (i) 
        # (Message flows j -> i)
        
        src_idx = edge_indices[:, 0]
        dst_idx = edge_indices[:, 1]
        
        atom_i = atom_features[dst_idx] # Target
        atom_j = atom_features[src_idx] # Source
        
        # 2. Concatenate [vi, vj, u_ij]
        # Shape: (N_edges, 2*node_dim + edge_dim)
        z_ij = torch.cat([atom_i, atom_j, edge_features], dim=1)
        
        # 3. Compute Signal and Filter
        # Core signal
        signal = self.fc_core(z_ij)
        signal = self.bn_core(signal)
        signal = F.softplus(signal) # Non-linearity g(z)
        
        # Filter (Gate)
        gate = self.fc_filter(z_ij)
        gate = self.bn_filter(gate)
        gate = torch.sigmoid(gate) # Sigmoid sigma(z)
        
        # 4. Message = Gate * Signal
        message = gate * signal
        
        # 5. Aggregate Messages (Sum pooling)
        # We need to sum messages for each target atom (dst_idx)
        # Ideally use scatter_add from torch_scatter, but to keep dependencies minimal (pure torch):
        # We can implement a naive scatter add or use index_add_
        
        # Prepare output buffer
        nbr_sumed = torch.zeros_like(atom_features)
        
        # Add messages to target atoms
        # nbr_sumed[dst_idx] += message -> this is not in-place safe for duplicates in dst_idx
        # Use index_add_:
        nbr_sumed.index_add_(0, dst_idx, message)
        
        # 6. Update Step (Residual Connection)
        out = atom_features + nbr_sumed
        
        return out

if __name__ == "__main__":
    # Test Block
    print("Verifying CGCNNConv Layer...")
    
    # Dummy Params
    N_atoms = 5
    N_edges = 12 # 5 atoms connected somehow
    D_node = 64
    D_edge = 40
    
    conv = CGCNNConv(D_node, D_edge)
    
    # Dummy Data
    x = torch.randn(N_atoms, D_node)
    edge_attr = torch.randn(N_edges, D_edge)
    
    # Random edge indices (src -> dst)
    src = torch.randint(0, N_atoms, (N_edges,))
    dst = torch.randint(0, N_atoms, (N_edges,))
    edge_index = torch.stack([src, dst], dim=1)
    
    # Forward
    out = conv(x, edge_attr, edge_index)
    
    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {out.shape}")
    
    assert out.shape == (N_atoms, D_node)
    print("✅ CGCNNConv verified.")
