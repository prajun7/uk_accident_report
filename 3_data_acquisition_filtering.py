import os
import pandas as pd

def run():
    print("\n--- STEP 3: Data Acquisition & Filtering ---")
    
    # 1. Acquire data directly from the known local dataset directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    acc_path = os.path.join(base_dir, 'Accidents0515.csv')
    
    print(f"Loading accidents dataset from: {acc_path}")
    
    # We read a subset of columns or handle bad lines to filter broken data early
    # For speed in development, consider adding nrows=100000 if needed, but for big data we load all
    try:
        accidents = pd.read_csv(acc_path, low_memory=False, on_bad_lines='skip')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Original shape: {accidents.shape}")
    
    # 2. Filtering
    # Drop rows without our critical target variable 'Accident_Severity'
    if 'Accident_Severity' in accidents.columns:
        before_drop = len(accidents)
        accidents.dropna(subset=['Accident_Severity'], inplace=True)
        print(f"Dropped {before_drop - len(accidents)} rows lacking targets.")
        
    # Drop heavily corrupted or overwhelmingly null columns early to save memory
    threshold = 0.4 * len(accidents)
    null_counts = accidents.isnull().sum()
    drop_cols = null_counts[null_counts > threshold].index.tolist()
    
    if drop_cols:
        print(f"Filtering out columns with >40% missing data: {drop_cols}")
        accidents.drop(columns=drop_cols, inplace=True)

    # 3. Save acquired and filtered baseline
    out_path = os.path.join(output_dir, '3_filtered_accidents.csv')
    accidents.to_csv(out_path, index=False)
    print(f"Saved filtered base to {out_path} with shape {accidents.shape}")

if __name__ == "__main__":
    run()
