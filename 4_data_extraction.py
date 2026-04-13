import os
import pandas as pd

def run():
    print("\n--- STEP 4: Data Extraction ---")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'output')
    in_path = os.path.join(output_dir, '3_filtered_accidents.csv')
    
    if not os.path.exists(in_path):
        print(f"Error: Could not find {in_path}. Did Step 3 run?")
        return

    print(f"Loading filtered base from {in_path}")
    df = pd.read_csv(in_path, low_memory=False)

    # We extract columns specific to our business case predicting severity prior to crashes:
    # 1. Target: Accident_Severity
    # 2. Environmental Pre-Crash Variables: Speed_limit, Road_Type, Light_Conditions, 
    #    Weather_Conditions, Road_Surface_Conditions, Time, Date, Day_of_Week, Urban_or_Rural_Area
    
    required_cols = [
        'Accident_Severity',
        'Speed_limit', 'Road_Type', 'Light_Conditions', 
        'Weather_Conditions', 'Road_Surface_Conditions',
        'Time', 'Date', 'Day_of_Week', 'Urban_or_Rural_Area',
        'Number_of_Vehicles' # Often known context
    ]

    # Check which of these actually exist inside the DataFrame
    keep_cols = [c for c in required_cols if c in df.columns]
    
    print(f"Extracting vital feature columns: {keep_cols}")
    extracted_df = df[keep_cols].copy()
    
    out_path = os.path.join(output_dir, '4_extracted_data.csv')
    extracted_df.to_csv(out_path, index=False)
    print(f"Extracted dataset saved to {out_path} with shape {extracted_df.shape}")

if __name__ == "__main__":
    run()
