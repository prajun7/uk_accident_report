import os
import pandas as pd
import numpy as np

def run():
    print("\n--- STEP 5: Data Validation & Cleansing ---")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'output')
    in_path = os.path.join(output_dir, '4_extracted_data.csv')
    
    if not os.path.exists(in_path):
        print(f"Error: Could not find {in_path}.")
        return

    df = pd.read_csv(in_path, low_memory=False)
    print(f"Loaded extracted data with shape {df.shape}")

    # 1. Replace the DfT standard placeholder -1 with NaN representing missing data
    print("Replacing DfT standard placeholder '-1' with system NaN.")
    df.replace(-1, np.nan, inplace=True)
    
    # 2. Imputation of Nulls
    print(f"Nulls before imputation: {df.isnull().sum().sum()}")
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64, float, int]:
            # Fill numeric with median
            df[col] = df[col].fillna(df[col].median())
        else:
            # Fill categoricals with mode
            mode_series = df[col].mode()
            if len(mode_series) > 0:
                df[col] = df[col].fillna(mode_series[0])
                
    print(f"Nulls after imputation: {df.isnull().sum().sum()}")

    # 3. Cap heavy outliers
    if 'Speed_limit' in df.columns:
        valid_limits = [10, 20, 30, 40, 50, 60, 70]
        # Any speed limit not standard might be noise. We map extreme noise to the max valid limit.
        df['Speed_limit'] = df['Speed_limit'].apply(lambda x: x if x in valid_limits else 70)
        print("Validated Speed Limits.")
        
    out_path = os.path.join(output_dir, '5_cleansed_data.csv')
    df.to_csv(out_path, index=False)
    print(f"Cleansed data saved to {out_path} with shape {df.shape}")

if __name__ == "__main__":
    run()
