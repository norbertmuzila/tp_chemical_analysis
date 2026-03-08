"""
Plot reflectance vs wavelength for a single mineral with all available samples.
Shows both individual spectra and mean curve, labeled with mineral name and data type (AREF/RREF).
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def extract_mineral_and_type(filename: str):
    """Extract mineral name and data type (AREF/RREF) from filename."""
    parts = filename.split("_")
    if len(parts) >= 3:
        mineral = parts[2]
    else:
        mineral = "Unknown"
    
    # Check for AREF or RREF in filename
    if "AREF" in filename:
        data_type = "AREF"
    elif "RREF" in filename:
        data_type = "RREF"
    else:
        data_type = "Other"
    
    return mineral, data_type


def main(xlsx_path, mineral_name):
    """
    Load data and plot all spectra for the specified mineral.
    
    Args:
        xlsx_path: path to Combined_ASD_Data.xlsx
        mineral_name: name of mineral to plot (e.g. "Olivine", "Muscovite")
    """
    
    # Load data
    df = pd.read_excel(xlsx_path, sheet_name="Reflectance_vs_Wavelength")
    wavelengths = df["Wavelength"].values
    
    # Find all columns for this mineral
    mineral_cols = []
    aref_cols = []
    rref_cols = []
    
    for col in df.columns[1:]:
        m, dtype = extract_mineral_and_type(col)
        if m == mineral_name:
            mineral_cols.append(col)
            if dtype == "AREF":
                aref_cols.append(col)
            elif dtype == "RREF":
                rref_cols.append(col)
    
    if not mineral_cols:
        print(f"Mineral '{mineral_name}' not found in data.")
        print("Available minerals:")
        for col in df.columns[1:]:
            m, _ = extract_mineral_and_type(col)
            print(f"  {m}")
        return
    
    print(f"Found {len(mineral_cols)} samples for {mineral_name}")
    print(f"  - AREF: {len(aref_cols)} samples")
    print(f"  - RREF: {len(rref_cols)} samples")
    
    # Create subplots: one for AREF, one for RREF
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot AREF data
    if aref_cols:
        ax = axes[0]
        for col in aref_cols:
            spec = df[col].values
            ax.plot(wavelengths, spec, alpha=0.5, linewidth=0.8)
        
        # Plot mean
        aref_data = df[aref_cols].values
        mean_aref = np.nanmean(aref_data, axis=1)
        ax.plot(wavelengths, mean_aref, color='red', linewidth=2, label='Mean')
        
        ax.set_xlabel('Wavelength (µm)', fontsize=12)
        ax.set_ylabel('Reflectance', fontsize=12)
        ax.set_title(f'{mineral_name} - AREF (Absolute REFlectance)\n{len(aref_cols)} samples', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        axes[0].text(0.5, 0.5, 'No AREF data available', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title(f'{mineral_name} - AREF', fontsize=14, fontweight='bold')
    
    # Plot RREF data
    if rref_cols:
        ax = axes[1]
        for col in rref_cols:
            spec = df[col].values
            ax.plot(wavelengths, spec, alpha=0.5, linewidth=0.8)
        
        # Plot mean
        rref_data = df[rref_cols].values
        mean_rref = np.nanmean(rref_data, axis=1)
        ax.plot(wavelengths, mean_rref, color='blue', linewidth=2, label='Mean')
        
        ax.set_xlabel('Wavelength (µm)', fontsize=12)
        ax.set_ylabel('Reflectance', fontsize=12)
        ax.set_title(f'{mineral_name} - RREF (Relative REFlectance)\n{len(rref_cols)} samples', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        axes[1].text(0.5, 0.5, 'No RREF data available', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title(f'{mineral_name} - RREF', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    out_dir = Path("analysis_outputs")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{mineral_name}_reflectance_vs_wavelength.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {out_file}")
    
    # Also save the mean spectra to CSV
    csv_file = out_dir / f"{mineral_name}_spectra.csv"
    csv_data = {"Wavelength": wavelengths}
    
    if aref_cols:
        aref_data = df[aref_cols].values
        csv_data["AREF_Mean"] = np.nanmean(aref_data, axis=1)
        for i, col in enumerate(aref_cols):
            csv_data[f"AREF_{i+1}"] = df[col].values
    
    if rref_cols:
        rref_data = df[rref_cols].values
        csv_data["RREF_Mean"] = np.nanmean(rref_data, axis=1)
        for i, col in enumerate(rref_cols):
            csv_data[f"RREF_{i+1}"] = df[col].values
    
    csv_df = pd.DataFrame(csv_data)
    csv_df.to_csv(csv_file, index=False)
    print(f"Data saved to: {csv_file}")
    
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default: use Olivine (most common mineral)
        mineral = "Olivine"
        xlsx_path = "uploads/Combined_ASD_Data.xlsx"
        print(f"No mineral specified. Defaulting to: {mineral}")
        print(f"Usage: python3 plot_single_mineral.py <mineral_name> [xlsx_path]")
    else:
        mineral = sys.argv[1]
        xlsx_path = sys.argv[2] if len(sys.argv) > 2 else "uploads/Combined_ASD_Data.xlsx"
    
    main(xlsx_path, mineral)
