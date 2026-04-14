import os
import pandas as pd

PIPELINE_SOURCE_COLS = [
    'Accident_Severity',
    # Spatial fields retained to build Risk_Zone in Step 6, then dropped before final ML.
    'Latitude', 'Longitude',
    'Local_Authority_(District)', 'Police_Force',
    # Road network structure
    'Speed_limit', 'Road_Type',
    '1st_Road_Class', '1st_Road_Number',
    '2nd_Road_Class', '2nd_Road_Number',
    'Junction_Detail', 'Junction_Control',
    # Environment / scene conditions
    'Light_Conditions', 'Weather_Conditions',
    'Road_Surface_Conditions', 'Special_Conditions_at_Site',
    'Carriageway_Hazards',
    'Pedestrian_Crossing-Human_Control',
    'Pedestrian_Crossing-Physical_Facilities',
    # Temporal context used to derive Hour / IsNight / cyclical features
    'Time', 'Day_of_Week',
    # Area context
    'Urban_or_Rural_Area',
    # Counts retained for clustering-side context and dropped before final model fit
    'Number_of_Casualties', 'Number_of_Vehicles',
]

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

    # This is the schema handoff into the current risk-zone workflow:
    # keep only the columns needed for cleansing, spatial clustering, and
    # engineered-feature creation. Date/month fields are intentionally omitted
    # because the active plan uses hour-of-day and day-of-week instead.
    keep_cols = [c for c in PIPELINE_SOURCE_COLS if c in df.columns]
    missing_cols = [c for c in PIPELINE_SOURCE_COLS if c not in df.columns]

    if missing_cols:
        print(f"Warning: source file is missing expected pipeline columns: {missing_cols}")
    
    print(f"Extracting {len(keep_cols)} pipeline columns: {keep_cols}")
    extracted_df = df[keep_cols].copy()
    
    out_path = os.path.join(output_dir, '4_extracted_data.csv')
    extracted_df.to_csv(out_path, index=False)
    print(f"Extracted dataset saved to {out_path} with shape {extracted_df.shape}")

if __name__ == "__main__":
    run()
