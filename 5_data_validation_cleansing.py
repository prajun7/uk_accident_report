import os
import numpy as np
import pandas as pd

SENTINEL_NEG1_COLS = {
    '1st_Road_Number',
    '2nd_Road_Class',
    '2nd_Road_Number',
    'Junction_Detail',
    'Junction_Control',
    'Weather_Conditions',
    'Road_Surface_Conditions',
    'Special_Conditions_at_Site',
    'Carriageway_Hazards',
    'Pedestrian_Crossing-Human_Control',
    'Pedestrian_Crossing-Physical_Facilities',
}

PRESERVE_NULL_COLS = {'Latitude', 'Longitude', 'Time'}

MODE_IMPUTE_COLS = {
    'Accident_Severity',
    'Speed_limit',
    'Road_Type',
    '1st_Road_Class',
    '1st_Road_Number',
    '2nd_Road_Class',
    '2nd_Road_Number',
    'Junction_Detail',
    'Junction_Control',
    'Light_Conditions',
    'Weather_Conditions',
    'Road_Surface_Conditions',
    'Special_Conditions_at_Site',
    'Carriageway_Hazards',
    'Pedestrian_Crossing-Human_Control',
    'Pedestrian_Crossing-Physical_Facilities',
    'Day_of_Week',
    'Urban_or_Rural_Area',
}

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

    # 1. Replace sentinel -1 only on DfT coded columns where it means missing/unknown.
    print("Replacing DfT sentinel '-1' only on known coded columns.")
    sentinel_counts = {}
    for col in sorted(SENTINEL_NEG1_COLS):
        if col in df.columns:
            count = int((df[col] == -1).sum())
            if count > 0:
                df.loc[df[col] == -1, col] = np.nan
            sentinel_counts[col] = count
    print(f"Sentinel replacements: {sentinel_counts}")
    
    # 2. Imputation of Nulls
    print(f"Nulls before imputation: {df.isnull().sum().sum()}")
    for col in df.columns:
        if col in PRESERVE_NULL_COLS:
            # Preserve real missingness for geography/time so downstream steps can
            # decide whether to drop or derive these rows explicitly.
            continue

        if df[col].isnull().sum() == 0:
            continue

        if col in MODE_IMPUTE_COLS:
            mode_series = df[col].mode(dropna=True)
            if len(mode_series) > 0:
                df[col] = df[col].fillna(mode_series.iloc[0])
        elif df[col].dtype in [np.float64, np.int64, float, int]:
            df[col] = df[col].fillna(df[col].median())
        else:
            mode_series = df[col].mode(dropna=True)
            if len(mode_series) > 0:
                df[col] = df[col].fillna(mode_series.iloc[0])
                
    remaining_nulls = df.isnull().sum()
    print(f"Nulls after imputation/preservation: {int(remaining_nulls.sum())}")
    print("Remaining nulls by column:")
    print(remaining_nulls[remaining_nulls > 0].to_string())

    # 3. Validate speed limits without forcing bad values to 70 mph
    if 'Speed_limit' in df.columns:
        valid_limits = [10, 20, 30, 40, 50, 60, 70]
        invalid_mask = ~df['Speed_limit'].isin(valid_limits)
        invalid_count = int(invalid_mask.sum())
        if invalid_count > 0:
            df.loc[invalid_mask, 'Speed_limit'] = np.nan
            mode_speed = df['Speed_limit'].mode(dropna=True).iloc[0]
            df['Speed_limit'] = df['Speed_limit'].fillna(mode_speed)
        print(f"Validated Speed Limits. Replaced {invalid_count} invalid values using modal valid speed.")
        
    out_path = os.path.join(output_dir, '5_cleansed_data.csv')
    df.to_csv(out_path, index=False)
    print(f"Cleansed data saved to {out_path} with shape {df.shape}")

if __name__ == "__main__":
    run()
