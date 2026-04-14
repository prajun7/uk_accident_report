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

    # Environmental + Geographic pre-crash variables:
    # Lat/Long provide strong spatial signal for high-risk zones.
    # Months dropped — temporal granularity within year adds noise not signal.
    required_cols = [
        'Accident_Severity',
        # Geographic (strongest predictors)
        'Latitude', 'Longitude',
        'Local_Authority_(District)', 'Police_Force',
        # Road Characteristics
        'Speed_limit', 'Road_Type',
        '1st_Road_Class', '1st_Road_Number',
        '2nd_Road_Class', '2nd_Road_Number',
        'Junction_Detail', 'Junction_Control',
        # Environment
        'Light_Conditions', 'Weather_Conditions',
        'Road_Surface_Conditions', 'Special_Conditions_at_Site',
        'Carriageway_Hazards',
        'Pedestrian_Crossing-Human_Control',
        'Pedestrian_Crossing-Physical_Facilities',
        # Temporal (time of day, not month)
        'Time', 'Day_of_Week',
        # Area type
        'Urban_or_Rural_Area',
        # Lookup-table source columns (dropped before training but used for aggregation)
        'Number_of_Casualties', 'Number_of_Vehicles',
    ]

    # Check which of these actually exist inside the DataFrame
    keep_cols = [c for c in required_cols if c in df.columns]
    
    print(f"Extracting {len(keep_cols)} feature columns: {keep_cols}")
    extracted_df = df[keep_cols].copy()
    
    out_path = os.path.join(output_dir, '4_extracted_data.csv')
    extracted_df.to_csv(out_path, index=False)
    print(f"Extracted dataset saved to {out_path} with shape {extracted_df.shape}")

if __name__ == "__main__":
    run()
