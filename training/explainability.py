import torch
import numpy as np
import logging
from typing import Dict, List, Tuple
from pymatgen.core import Structure

from training.inference import CGCNNPredictor

logger = logging.getLogger(__name__)

class CGCNNExplainer:
    """
    Interpretability module for CGCNN.
    
    Explains model predictions by identifying which atoms contributed most
    to the output property.
    
    Method: Gradient-based Saliency (Input Gradients)
    -------------------------------------------------
    We compute the gradient of the predicted property y w.r.t the atomic features X.
    Importance_i = || d(y)/d(X_i) ||_2
    
    This answers: "How much would the prediction change if we perturbed this atom's features?"
    """
    
    def __init__(self, predictor: CGCNNPredictor):
        self.predictor = predictor
        
    def explain(self, structure: Structure, target_index: int = 0) -> Dict[str, any]:
        """
        Compute atomic importance scores.
        
        Args:
            structure: Input structure.
            target_index: Index of the output property to explain (0: FormE, 1: BandGap, 2: Density).
            
        Returns:
            Dict containing:
            - 'atomic_importance': List of scores, normalized 0-1.
            - 'most_influential_atom': Element symbol of max score.
        """
        # 1. Prepare Inputs
        # We need to manually do what `predictor.predict` does but keep gradients enabled
        
        # Build graph
        atom_fea, nbr_fea, nbr_idx = self.predictor.builder.get_graph(structure)
        n_atoms = atom_fea.shape[0]
        batch_mapping = torch.zeros(n_atoms, dtype=torch.long)
        
        # Move to device
        atom_fea = atom_fea.to(self.predictor.device)
        nbr_fea = nbr_fea.to(self.predictor.device)
        nbr_idx = nbr_idx.to(self.predictor.device)
        batch_mapping = batch_mapping.to(self.predictor.device)
        
        # Enable Gradients for Feature Input
        # Note: atom_fea is (N_atoms, node_dim). Saliency is gradient w.r.t this tensor.
        atom_fea.requires_grad = True
        
        # 2. Forward Pass
        self.predictor.model.eval() # Keep eval mode (dropout off, batchnorm fixed)
        
        # Forward
        preds_norm = self.predictor.model(atom_fea, nbr_fea, nbr_idx, batch_mapping)
        
        # Select Target
        target_prediction = preds_norm[0, target_index]
        
        # 3. Backward Pass (Saliency)
        # We want d(target) / d(atom_fea)
        
        # Zero gradients
        self.predictor.model.zero_grad()
        
        # Compute gradient
        target_prediction.backward()
        
        # Get gradient: Shape (N_atoms, node_dim)
        gradients = atom_fea.grad
        
        if gradients is None:
             logger.warning("No gradients computed.")
             return {"atomic_importance": [0.0]*n_atoms}
             
        gradients = gradients.cpu().numpy()
        
        # 4. Compute Importance Score
        # L2 norm across feature dimension -> Reduces (N_atoms, Feat_Dim) to (N_atoms,)
        # Suggests overall sensitivity of the atom
        importance_scores = np.linalg.norm(gradients, axis=1)
        
        # Normalize to 0-1 for visualization
        if importance_scores.max() > 0:
            importance_scores = importance_scores / importance_scores.max()
            
        # 5. Map to Atoms
        # pymatgen structure sites correspond to rows 0..N-1
        elements = [site.specie.symbol for site in structure.sites]
        
        max_idx = np.argmax(importance_scores)
        most_influential = elements[max_idx]
        
        return {
            "atomic_importance": importance_scores.tolist(),
            "elements": elements,
            "most_influential_atom_index": int(max_idx),
            "most_influential_element": most_influential
        }

if __name__ == "__main__":
    # Test Block
    print("Testing Explainability...")
    import os
    
    # Mock Predictor Setup
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    model_path = os.path.join(base_dir, "models", "cgcnn_best.pt")
    
    if os.path.exists(model_path):
        predictor = CGCNNPredictor(model_path, device="cpu")
        explainer = CGCNNExplainer(predictor)
        
        # Create Dummy Perovskite
        from pymatgen.core import Lattice
        # Simple cubic SrTiO3-like arrangement
        l = Lattice.cubic(3.905)
        # Sr at corner, Ti at center, O at faces
        sites = ["Sr", "Ti", "O", "O", "O"]
        coords = [
            [0,0,0],
            [0.5,0.5,0.5],
            [0.5,0.5,0],
            [0.5,0,0.5],
            [0,0.5,0.5]
        ]
        struct = Structure(l, sites, coords)
        
        # Explain Band Gap (Target 1)
        print("\nExplaining Band Gap Prediction for SrTiO3...")
        result = explainer.explain(struct, target_index=1)
        
        print("Importance Scores:")
        for el, score in zip(result['elements'], result['atomic_importance']):
            print(f"  > {el}: {score:.4f}")
            
        print(f"\nMost Influential: {result['most_influential_element']} (Index {result['most_influential_atom_index']})")
        print("\nInterpretation: This atom has the highest gradient magnitude, meaning changing its properties impacts the band gap the most.")
    else:
        print("Skipping run (No model found).")
