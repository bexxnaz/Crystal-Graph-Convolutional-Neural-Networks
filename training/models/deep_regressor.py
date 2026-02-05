import torch
import torch.nn as nn
import torch.nn.init as init

class DeepMaterialRegressor(nn.Module):
    """
    Deep Neural Network for multi-target material property prediction.
    
    Predicts:
    1. Formation Energy per Atom (eV/atom)
    2. Band Gap (eV)
    3. Density (g/cm^3)
    
    Architecture:
    - Input: Composition-based feature vector (dim=20)
    - Hidden: 5 Fully Connected Layers with formatting [Input -> 128 -> 256 -> 128 -> 64 -> Output]
    - Regularization: Batch Normalization and Dropout (p=0.3)
    - Activation: LeakyReLU (for better gradient flow than standard ReLU)
    """
    def __init__(self, input_dim: int = 20, output_dim: int = 3, dropout_prob: float = 0.3):
        super(DeepMaterialRegressor, self).__init__()
        
        # 1. Define Layers
        # Layer 1: Expansion
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        
        # Layer 2: Deep Expansion
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        # Layer 3: Bottleneck 1
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Layer 4: Bottleneck 2
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        
        # Layer 5: Output
        self.fc_out = nn.Linear(64, output_dim)
        
        # Activation & Regularization
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(p=dropout_prob)
        
        # 2. Weight Initialization
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Block 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Block 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Block 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Block 4
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Output Head (No activation, regression task)
        out = self.fc_out(x)
        return out

    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract latent material embeddings from the penultimate layer (fc4).
        
        Args:
            x: Input features.
            
        Returns:
            Tensor of shape (batch, 64) representing the learned embedding.
        """
        # Block 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Block 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Block 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Block 4
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.activation(x)
        # Note: We typically take embeddings *after* activation but *before* dropout for inference analysis
        # No dropout here as this is for representation extraction
        
        return x

    def _initialize_weights(self):
        """Apply Kaiming Initialization for ReLU/LeakyReLU activations."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He (Kaiming) Normal initialization
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

if __name__ == "__main__":
    # Verification Block
    print("Verifying DeepMaterialRegressor architecture...")
    model = DeepMaterialRegressor()
    dummy_input = torch.randn(5, 20) # Batch size=5, Feature dim=20
    output = model(dummy_input)
    embeddings = model.extract_embeddings(dummy_input)
    
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")
    print(f"Embedding Shape: {embeddings.shape}") # Should be (5, 64)
    print("Architecture verified.")
