import pandas as pd
import numpy as np
import os

# Paths
TRAIN_PATH = "D:/coding_files/Projects/matterGen/material dataset/train.parquet"

def inspect_data():
    if not os.path.exists(TRAIN_PATH):
        print(f"File not found: {TRAIN_PATH}")
        return

    print("Loading dataset...")
    df = pd.read_parquet(TRAIN_PATH)
    
    print(f"\nTotal Samples: {len(df)}")
    print("-" * 30)
    
    # 1. Band Gap Analysis
    bg = df['dft_band_gap']
    n_metals = (bg < 0.01).sum()
    print(f"\n[Band Gap Analysis]")
    print(f"Total entries: {len(bg)}")
    print(f"Metals (Gap < 0.01 eV): {n_metals} ({n_metals/len(df)*100:.1f}%)")
    print(f"Insulators (Gap >= 0.01 eV): {len(df) - n_metals}")
    print(f"Mean Gap (All): {bg.mean():.4f} eV")
    print(f"Mean Gap (Insulators only): {bg[bg >= 0.01].mean():.4f} eV")
    print(f"Max Gap: {bg.max():.4f} eV")
    
    # 2. Stability Analysis
    e_hull = df['energy_above_hull']
    print(f"\n[Stability (Energy Above Hull) Analysis]")
    print(f"Stable materials (Energy <= 0.05): {(e_hull <= 0.05).sum()} ({(e_hull <= 0.05).sum()/len(df)*100:.1f}%)")
    print(f"Unstable outliers (Energy > 5.0): {(e_hull > 5.0).sum()}")
    print(f"Mean Energy: {e_hull.mean():.4f}")
    print(f"Std Dev: {e_hull.std():.4f}")

    # 3. Missing Data
    print(f"\n[Missing Values]")
    print(df[['positions', 'cell', 'atomic_numbers', 'dft_band_gap', 'energy_above_hull']].isnull().sum())

if __name__ == "__main__":
    inspect_data()
