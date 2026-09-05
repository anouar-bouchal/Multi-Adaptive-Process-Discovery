# convert_log.py

from pathlib import Path
import pandas as pd
import pm4py

def convert_xes_log():
    """
    Reads the 'BPI Challenge 2017.xes' event log from the script's directory,
    converts it to a Pandas DataFrame, and saves it in Pickle, Excel, and CSV formats.
    """
    # --- 1. Define File Paths ---
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.resolve()
    
    # Define the input XES file path
    xes_file_path = script_dir / "BPI Challenge 2017.xes"
    
    # Define output file paths using the same base name
    base_name = xes_file_path.stem
    pkl_output_path = script_dir / f"{base_name}.pkl"
    xlsx_output_path = script_dir / f"{base_name}.xlsx"
    csv_output_path = script_dir / f"{base_name}.csv"

    # --- 2. Check if Input File Exists ---
    if not xes_file_path.exists():
        print(f"ERROR: Input file not found at '{xes_file_path}'")
        print("Please make sure 'BPI Challenge 2017.xes' is in the same directory as this script.")
        return # Exit the function

    print(f"Found input file: '{xes_file_path.name}'")

    try:
        # --- 3. Read and Convert the Log ---
        print("\nReading XES file... (This may take a few minutes for a large file like BPI 2017)")
        log = pm4py.read_xes(str(xes_file_path))
        print("XES file read successfully.")

        print("\nConverting event log to Pandas DataFrame...")
        dataframe = pm4py.convert_to_dataframe(log)
        print("DataFrame conversion complete.")
        print("-" * 30)
        print("DataFrame Info:")
        print(f"  - Shape: {dataframe.shape}")
        print(f"  - Columns: {dataframe.columns.tolist()}")
        print("-" * 30)
        print("DataFrame Head (first 5 rows):")
        print(dataframe.head())
        print("-" * 30)

        # --- 4. Save to Different Formats ---
        print("\n--- Starting file export ---")

        # Save to Pickle (.pkl) - FASTEST
        print(f"\n[1/3] Saving to Pickle format -> '{pkl_output_path.name}'...")
        dataframe.to_pickle(pkl_output_path)
        print("Saved to Pickle.")

        # Save to CSV (.csv) - RELATIVELY FAST
        print(f"\n[2/3] Saving to CSV format -> '{csv_output_path.name}'...")
        dataframe.to_csv(csv_output_path, index=False)
        print("Saved to CSV.")
        
        # Save to Excel (.xlsx) - VERY SLOW AND MEMORY INTENSIVE
        print(f"\n[3/3] Saving to Excel format -> '{xlsx_output_path.name}'...")
        print("      ⚠️ WARNING: This step can be very slow and consume a lot of memory for large files.")
        dataframe.to_excel(xlsx_output_path, index=False, engine='openpyxl')
        print("Saved to Excel.")

        print("\n\n--- All operations complete! ---")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        print("Please ensure you have installed the required libraries with: pip install pm4py pandas openpyxl")

if __name__ == "__main__":
    convert_xes_log()
