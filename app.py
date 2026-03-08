"""
Flask web app for exploring mineral spectra from the ASD catalogue.
Provides an interactive browser interface to view plots and spectral data.
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from pathlib import Path
import json

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Load data on startup
XLSX_PATH = "uploads/Combined_ASD_Data.xlsx"
OUTPUT_DIR = Path("analysis_outputs")

def get_available_minerals():
    """Get list of all minerals in the dataset."""
    df = pd.read_excel(XLSX_PATH, sheet_name="Summary")
    minerals = []
    seen = set()
    
    for filename in df['Filename']:
        parts = filename.split("_")
        if len(parts) >= 3:
            mineral = parts[2]
            if mineral not in seen:
                minerals.append(mineral)
                seen.add(mineral)
    
    return sorted(set(minerals))

def get_mineral_stats(mineral_name):
    """Get statistics for a specific mineral."""
    df = pd.read_excel(XLSX_PATH, sheet_name="Summary")
    mask = df['Filename'].str.contains(f"_{mineral_name}_", regex=False)
    mineral_data = df[mask]
    
    stats = {
        'name': mineral_name,
        'total_samples': len(mineral_data),
        'mean_reflectance': mineral_data['Mean_Value'].mean(),
        'min_reflectance': mineral_data['Min_Value'].min(),
        'max_reflectance': mineral_data['Max_Value'].max(),
        'aref_count': sum(1 for f in mineral_data['Filename'] if 'AREF' in f),
        'rref_count': sum(1 for f in mineral_data['Filename'] if 'RREF' in f),
    }
    return stats

@app.route('/')
def index():
    """Home page with mineral selector."""
    minerals = get_available_minerals()
    return render_template('index.html', minerals=minerals)

@app.route('/api/minerals')
def api_minerals():
    """API endpoint to get all minerals."""
    minerals = get_available_minerals()
    return jsonify(minerals)

@app.route('/api/mineral/<mineral_name>')
def api_mineral_stats(mineral_name):
    """API endpoint for mineral statistics."""
    try:
        stats = get_mineral_stats(mineral_name)
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/plot/<mineral_name>')
def api_plot_url(mineral_name):
    """Get URL to mineral plot image."""
    plot_file = OUTPUT_DIR / f"{mineral_name}_reflectance_vs_wavelength.png"
    if plot_file.exists():
        return jsonify({'plot_url': f'/plots/{mineral_name}_reflectance_vs_wavelength.png'})
    else:
        return jsonify({'error': 'Plot not found'}), 404

@app.route('/plots/<filename>')
def get_plot(filename):
    """Serve plot images."""
    from flask import send_from_directory
    return send_from_directory(OUTPUT_DIR, filename)

@app.route('/data/<filename>')
def get_data(filename):
    """Serve CSV data files."""
    from flask import send_from_directory
    return send_from_directory(OUTPUT_DIR, filename)

@app.route('/olivine')
def olivine():
    """Dedicated page for Olivine mineral analysis."""
    return render_template('olivine.html')

if __name__ == '__main__':
    print("🚀 Starting mineral spectroscopy web app...")
    print("📊 Open your browser at: http://localhost:5000")
    print("📈 Viewer at: http://localhost:5000/viewer")
    app.run(debug=True, host='0.0.0.0', port=5000)
