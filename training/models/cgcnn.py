import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from training.models.layers import CGCNNConv
from typing import Tuple


class CGCNN(nn.Module):
    """
    Crystal Graph Convolutional Neural Network (CGCNN).

    Architecture:
    1. Input atom features -> Embedding (Linear)
    2. N x CGCNN Convolution Layers (Message Passing)
    3. Crystal Pooling (Global Mean/Sum)
    4. Regression Heads (Output)
    """

    def __init__(
        self,
        node_input_dim: int = 4,
        edge_input_dim: int = 41,  # 8.0/0.2 + 1
        node_hidden_dim: int = 64,
        n_conv_layers: int = 3,
        n_targets: int = 3,
    ):
        """
        Args:
            node_input_dim: Raw node feature dim (Z, Group, Period, X).
            edge_input_dim: Raw edge feature dim (Gaussian distance).
            node_hidden_dim: Hidden dimension for atom features.
            n_conv_layers: Number of message passing steps.
            n_targets: Number of output properties.
        """
        super(CGCNN, self).__init__()

        # 1. Embedding
        # Project raw node features to hidden space
        self.embedding = nn.Linear(node_input_dim, node_hidden_dim)

        # 2. Convolutions
        self.convs = nn.ModuleList(
            [CGCNNConv(node_hidden_dim, edge_input_dim) for _ in range(n_conv_layers)]
        )

        # 3. Output Head
        # We start from the pooled crystal vector (same dim as node hidden)
        self.fc_out = nn.Sequential(
            nn.Linear(node_hidden_dim, node_hidden_dim),
            nn.BatchNorm1d(node_hidden_dim),
            nn.Softplus(),
            nn.Linear(node_hidden_dim, n_targets),
        )

        self.pooling = "mean"  # Or sum
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def pooling_layer(
        self, atom_features: torch.Tensor, batch_indices: torch.Tensor, n_graphs: int
    ) -> torch.Tensor:
        """
        Aggregate atom features into crystal features.

        Args:
            atom_features: (N_total_atoms, hidden_dim)
            batch_indices: (N_total_atoms,) mapping atoms to graph idx
            n_graphs: Batch size

        Returns:
            crystal_features: (n_graphs, hidden_dim)
        """
        device = atom_features.device
        crystal_features = torch.zeros(n_graphs, atom_features.size(1), device=device)

        # Sum pooling
        crystal_features.index_add_(0, batch_indices, atom_features)

        if self.pooling == "mean":
            # Compute counts
            counts = torch.zeros(n_graphs, device=device)
            ones = torch.ones(atom_features.size(0), device=device)
            counts.index_add_(0, batch_indices, ones)

            # Avoid division by zero
            counts = torch.clamp(counts, min=1.0)
            crystal_features = crystal_features / counts.unsqueeze(1)

        return crystal_features

    def get_crystal_embedding(
        self,
        atom_fea: torch.Tensor,
        nbr_fea: torch.Tensor,
        nbr_idx: torch.Tensor,
        batch_mapping: torch.Tensor,
        n_graphs: int,
    ) -> torch.Tensor:
        """
        Run the network and return the crystal-level embedding vector.
        Useful for visualization.
        """
        # 1. Embed
        x = self.embedding(atom_fea)

        # 2. Convolve
        for conv in self.convs:
            x = conv(x, nbr_fea, nbr_idx)

        # 3. Pool
        crys_fea = self.pooling_layer(x, batch_mapping, n_graphs)

        return crys_fea

    def forward(
        self,
        atom_fea: torch.Tensor,
        nbr_fea: torch.Tensor,
        nbr_idx: torch.Tensor,
        batch_mapping: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            atom_fea: (N_atoms, node_in)
            nbr_fea: (N_edges, edge_in)
            nbr_idx: (N_edges, 2)
            batch_mapping: (N_atoms,)
        """
        # Determine batch size from max index
        n_graphs = batch_mapping.max().item() + 1

        # Get pooled representation
        crys_fea = self.get_crystal_embedding(
            atom_fea, nbr_fea, nbr_idx, batch_mapping, n_graphs
        )

        # Predict
        out = self.fc_out(crys_fea)
        return out


if __name__ == "__main__":
    # Test Block
    print("Verifying CGCNN Model...")

    # Dummy Batch
    # Graph 1: 3 atoms, 6 edges | Graph 2: 2 atoms, 2 edges
    # Total: 5 atoms, 8 edges

    node_in = 4
    edge_in = 41
    hidden = 64

    model = CGCNN(
        node_input_dim=node_in, edge_input_dim=edge_in, node_hidden_dim=hidden
    )

    atom_fea = torch.randn(5, node_in)
    nbr_fea = torch.randn(8, edge_in)
    nbr_idx = torch.randint(0, 5, (8, 2))

    # Batch mapping: Atoms 0,1,2 -> Graph 0; Atoms 3,4 -> Graph 1
    batch_map = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)

    # Forward check
    out = model(atom_fea, nbr_fea, nbr_idx, batch_map)
    emb = model.get_crystal_embedding(atom_fea, nbr_fea, nbr_idx, batch_map, 2)

    print(f"Output Shape: {out.shape}")  # (2, 3)
    print(f"Embedding Shape: {emb.shape}")  # (2, 64)

    assert out.shape == (2, 3)
    assert emb.shape == (2, 64)
    print("✅ CGCNN Model Verified.")
