import os
import pandas as pd

PIPELINE_REQUIRED_COLS = [
    'Accident_Severity',
    'Latitude', 'Longitude',
    'Local_Authority_(District)', 'Police_Force',
    'Speed_limit', 'Road_Type',
    '1st_Road_Class', '1st_Road_Number',
    '2nd_Road_Class', '2nd_Road_Number',
    'Junction_Detail', 'Junction_Control',
    'Light_Conditions', 'Weather_Conditions',
    'Road_Surface_Conditions', 'Special_Conditions_at_Site',
    'Carriageway_Hazards',
    'Pedestrian_Crossing-Human_Control',
    'Pedestrian_Crossing-Physical_Facilities',
    'Time', 'Day_of_Week',
    'Urban_or_Rural_Area',
    'Number_of_Casualties', 'Number_of_Vehicles',
]

def run():
    print("\n--- STEP 3: Data Acquisition & Filtering ---")
    
    # 1. Acquire data directly from the known local dataset directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    acc_path = os.path.join(base_dir, 'Accidents0515.csv')
    
    print(f"Loading accidents dataset from: {acc_path}")
    
    # This stage is a coarse source-data sanity pass over the raw accidents file.
    # We keep the file mostly intact here and do the focused column selection in Step 4.
    try:
        accidents = pd.read_csv(acc_path, low_memory=False, on_bad_lines='skip')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Original shape: {accidents.shape}")
    
    # 2. Filtering
    # Accident_Severity is not the final modeling target anymore, but it remains
    # an important source quality label and is carried forward for audit/reference
    # until Step 6 derives the cluster-based Risk_Zone target.
    if 'Accident_Severity' in accidents.columns:
        before_drop = len(accidents)
        accidents.dropna(subset=['Accident_Severity'], inplace=True)
        print(f"Dropped {before_drop - len(accidents)} rows lacking Accident_Severity.")
        
    # Drop heavily corrupted non-pipeline columns early to save memory, but never
    # silently drop fields the downstream risk-zone pipeline depends on.
    threshold = 0.4 * len(accidents)
    null_counts = accidents.isnull().sum()
    high_null_cols = null_counts[null_counts > threshold].index.tolist()
    protected_high_null_cols = [c for c in high_null_cols if c in PIPELINE_REQUIRED_COLS]
    drop_cols = [c for c in high_null_cols if c not in PIPELINE_REQUIRED_COLS]
    
    if protected_high_null_cols:
        print(
            "Warning: pipeline-required columns exceeded the null threshold but were preserved: "
            f"{protected_high_null_cols}"
        )

    if drop_cols:
        print(f"Filtering out columns with >40% missing data: {drop_cols}")
        accidents.drop(columns=drop_cols, inplace=True)

    # 3. Save acquired and filtered baseline
    out_path = os.path.join(output_dir, '3_filtered_accidents.csv')
    accidents.to_csv(out_path, index=False)
    print(f"Saved filtered base to {out_path} with shape {accidents.shape}")

if __name__ == "__main__":
    run()
