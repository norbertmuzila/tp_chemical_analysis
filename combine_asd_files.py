import os
import glob
import pandas as pd
import re
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

def parse_asd_file(filepath):
    """Parse a single ASD text file and extract metadata and data."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            return None
        
        # Parse header
        header = lines[0].strip()
        
        # Extract information from filename for additional metadata
        filename = os.path.basename(filepath)
        
        # Parse the numeric values (reflectance data)
        data_values = []
        for line in lines[1:]:
            try:
                value = float(line.strip())
                data_values.append(value)
            except ValueError:
                continue
        
        return {
            'Filename': filename,
            'Header': header,
            'Data_Points': len(data_values),
            'Mean_Value': sum(data_values) / len(data_values) if data_values else None,
            'Min_Value': min(data_values) if data_values else None,
            'Max_Value': max(data_values) if data_values else None,
            'Data': data_values
        }
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        return None

def read_wavelength_file(filepath):
    """Read a wavelength text file and return a list of floats.

    The first line of the file is treated as a header and skipped; subsequent
    lines should each contain a single numeric value. Any non‑numeric lines
    are ignored.
    """
    values = []
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        for line in lines[1:]:
            try:
                val = float(line.strip())
                # ignore non‑positive wavelengths
                if val > 0:
                    values.append(val)
            except ValueError:
                # skip any stray text lines
                continue
    except Exception as e:
        print(f"Error reading wavelength file {filepath}: {e}")
    return values


def combine_asd_files():
    """Combine wavelengths and all ASD reflectance files into an Excel workbook."""

    upload_dir = '/workspaces/tp_chemical_analysis/uploads'

    # locate the wavelength file (there should only be one)
    wavelength_files = sorted(glob.glob(os.path.join(upload_dir, '*Wavelength*.txt')))
    wavelengths = []
    if wavelength_files:
        wl_path = wavelength_files[0]
        print(f"Using wavelength file: {os.path.basename(wl_path)}")
        wavelengths = read_wavelength_file(wl_path)
        print(f"Read {len(wavelengths)} wavelength points")
    else:
        print("No wavelength file found; wavelength column will be empty")

    # Find all ASD reflectance text files (exclude wavelength file itself)
    asd_files = sorted(glob.glob(os.path.join(upload_dir, 's07_ASD*.txt')))
    if wavelength_files:
        asd_files = [f for f in asd_files if f not in wavelength_files]

    print(f"Found {len(asd_files)} reflectance ASD files")

    # Parse all reflectance files
    all_data = []

    for i, filepath in enumerate(asd_files, 1):
        if i % 100 == 0:
            print(f"Processing file {i}/{len(asd_files)}...")
        parsed = parse_asd_file(filepath)
        if parsed:
            all_data.append(parsed)
    
    print(f"Successfully processed {len(all_data)} reflectance files")

    # Create Excel workbook
    output_file = os.path.join(upload_dir, 'Combined_ASD_Data.xlsx')

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Summary sheet
        summary_data = []
        for item in all_data:
            summary_data.append({
                'Filename': item['Filename'],
                'Header': item['Header'],
                'Data_Points': item['Data_Points'],
                'Mean_Value': item['Mean_Value'],
                'Min_Value': item['Min_Value'],
                'Max_Value': item['Max_Value']
            })
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)

        # create a combined sheet with wavelength + reflectance columns
        if wavelengths:
            combined = pd.DataFrame({'Wavelength': wavelengths})
            # ensure we have enough rows for the longest reflectance series
            max_len = len(wavelengths)
        else:
            combined = pd.DataFrame()
            max_len = 0

        for item in all_data:
            col_name = os.path.splitext(item['Filename'])[0]
            vals = item['Data']
            # negative reflectance are not physical; convert to NaN so they
            # don't contaminate averages
            vals = [v if v is None or v >= 0 else float('nan') for v in vals]
            # pad or trim to match wavelengths length
            if wavelengths:
                if len(vals) < max_len:
                    vals = vals + [None] * (max_len - len(vals))
                elif len(vals) > max_len:
                    vals = vals[:max_len]
            combined[col_name] = vals

        # write the combined dataset (will include wavelength column first if available)
        if not combined.empty:
            combined.to_excel(writer, sheet_name='Reflectance_vs_Wavelength', index=False)

        # full data sheet (original tabular form) for backwards compatibility
        max_points = max(len(item['Data']) for item in all_data) if all_data else 0
        full_data = []
        for item in all_data:
            row = {'Filename': item['Filename'], 'Header': item['Header']}
            for j, value in enumerate(item['Data']):
                row[f'Value_{j+1}'] = value
            for j in range(len(item['Data']), max_points):
                row[f'Value_{j+1}'] = None
            full_data.append(row)
        df_full = pd.DataFrame(full_data)
        df_full.to_excel(writer, sheet_name='Full_Data', index=False)

        print(f"\nExcel file created: {output_file}")
    
    return output_file

if __name__ == '__main__':
    output = combine_asd_files()
    print(f"Process complete! Output: {output}")
