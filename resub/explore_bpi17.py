# explore_bpi17.py

import pandas as pd
from pathlib import Path

def analyze_bpi17_log():
    """
    Loads the BPI 2017 pickle file and performs a quick analysis to inform
    the experimental setup for the research paper revision.
    """
    # --- 1. Load the Data ---
    script_dir = Path(__file__).parent.resolve()
    input_file = script_dir / "prelogs/BPI Challenge 2017.pkl"

    if not input_file.exists():
        print(f"❌ ERROR: Input file not found at '{input_file}'")
        print("Please run the 'convert_log.py' script first to generate the pickle file.")
        return

    print(f"✅ Found input file. Loading '{input_file.name}'...")
    try:
        df = pd.read_pickle(input_file)
        print("✅ DataFrame loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading pickle file: {e}")
        return

    # --- 2. Basic DataFrame Information ---
    print("\n" + "="*50)
    print("1. GENERAL DATAFRAME INFORMATION")
    print("="*50)
    # Using .info() gives a great summary of columns, non-null counts, and data types
    df.info(verbose=False)


    # --- 3. Analyze Lifecycle Transition ---
    lifecycle_col = 'lifecycle:transition'
    print(f"\n" + "="*50)
    print(f"2. ANALYSIS OF '{lifecycle_col}' COLUMN")
    print("="*50)
    if lifecycle_col in df.columns:
        print("This tells us which events mark the actual completion of a task.")
        print("We will likely filter the dataset to only include 'complete' events.")
        print("\nValue Counts:")
        print(df[lifecycle_col].value_counts())
    else:
        print(f"'{lifecycle_col}' column not found!")


    # --- 4. Analyze Potential Dimensions for Mini-Cubes ---
    print(f"\n" + "="*50)
    print("3. ANALYSIS OF POTENTIAL DIMENSIONAL COLUMNS")
    print("="*50)
    print("A good dimension has a small number of unique categories with a good distribution.")
    
    # List of candidate columns for the Mini-Cube dimension
    dimension_candidates = ['case:ApplicationType', 'case:LoanGoal']

    for col in dimension_candidates:
        if col in df.columns:
            print(f"\n--- Analyzing Column: '{col}' ---")
            
            # Check for missing values
            missing_count = df[col].isnull().sum()
            print(f"Missing Values: {missing_count} ({missing_count / len(df):.2%})")

            # Get the number of unique categories
            unique_count = df[col].nunique()
            print(f"Number of Unique Categories: {unique_count}")

            # Get the distribution of the top 10 categories
            print("\nDistribution of Top 10 Categories:")
            print(df[col].value_counts().nlargest(10))
        else:
            print(f"\n--- Column '{col}' not found! ---")

    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)


if __name__ == "__main__":
    analyze_bpi17_log()
