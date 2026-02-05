import os
import argparse
import logging
import subprocess
import sys

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_ensemble(args):
    """
    Train an ensemble of CGCNN models with different seeds.
    """
    logger.info(f"Training Ensemble of {args.n_models} models...")
    
    base_seed = args.start_seed
    
    for i in range(args.n_models):
        seed = base_seed + i
        model_name = f"cgcnn_seed_{seed}.pt"
        
        logger.info(f"--- Training Model {i+1}/{args.n_models} (Seed: {seed}) ---")
        
        # We call the train_cgcnn.py script as a subprocess to ensure clean state per run
        cmd = [
            sys.executable, "training/scripts/train_cgcnn.py",
            "--data_path", args.data_path,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--seed", str(seed),
            "--save_dir", args.save_dir,
            "--model_name", model_name,
            "--patience", str(args.patience)
        ]
        
        try:
            subprocess.check_check = True
            result = subprocess.run(cmd, check=True)
            logger.info(f"Model {i+1} completed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to train model {i+1}: {e}")
            sys.exit(1)

    logger.info("Ensemble training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CGCNN Ensemble")
    parser.add_argument("--n_models", type=int, default=5, help="Number of models in ensemble")
    parser.add_argument("--start_seed", type=int, default=42)
    parser.add_argument("--data_path", type=str, default="../../data/mock_materials.json")
    parser.add_argument("--epochs", type=int, default=50) # Shorter training for demo
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="../../models/ensemble")
    
    args = parser.parse_args()
    
    # Run from root of repo usually
    if not os.path.exists("training/scripts/train_cgcnn.py"):
        # try absolute path or relative
        # assuming running from mattergen-x root
        pass

    train_ensemble(args)
