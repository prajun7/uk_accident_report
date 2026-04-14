import os
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import LabelEncoder
import joblib

CORRELATION_FEATURE_COLS = [
    'Speed_limit',
    '1st_Road_Number',
    '2nd_Road_Number',
    'Day_of_Week',
    'Urban_or_Rural_Area',
    'Hour',
    'IsNight',
    'IsRushHour',
    'Hour_Sin',
    'Hour_Cos',
    'Urban_Speed_Net',
    'Junction_Complexity',
]

def run():
    print("\n--- STEP 6: Risk Zone Clustering & Aggregation ---")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'output')
    in_path = os.path.join(output_dir, '5_cleansed_data.csv')
    
    if not os.path.exists(in_path):
        print(f"Error: {in_path} not found.")
        return
        
    df = pd.read_csv(in_path, low_memory=False)

    # Missing coordinates cannot contribute to spatial clustering, so remove them
    # explicitly here instead of silently carrying them until Risk_Zone becomes NaN.
    missing_coords = int(df[['Latitude', 'Longitude']].isna().any(axis=1).sum())
    if missing_coords > 0:
        print(f"Dropping {missing_coords} rows with missing coordinates before clustering.")
        df = df.dropna(subset=['Latitude', 'Longitude']).copy()
    
    # 1. Unsupervised Learning: Spatial Hotspot Clustering
    print("Performing spatial clustering (Unsupervised Learning) to define Risk Zones...")
    coords = df[['Latitude', 'Longitude']]
    
    # Use MiniBatchKMeans to handle 1.7M rows quickly
    kmeans = MiniBatchKMeans(n_clusters=250, random_state=42, batch_size=10000, n_init='auto')
    df.loc[coords.index, 'Cluster'] = kmeans.fit_predict(coords)
    
    # Calculate density (accidents per cluster)
    cluster_counts = df['Cluster'].value_counts()
    
    # Define Risk Zones based on accident density per cluster with GAP ZONES
    # Low Risk: <35th percentile | Medium: 45th-75th | High: >85th
    low_thresh = cluster_counts.quantile(0.35)
    med_min = cluster_counts.quantile(0.45)
    med_max = cluster_counts.quantile(0.75)
    high_min = cluster_counts.quantile(0.85)
    
    def assign_risk(count):
        if count >= high_min: return 'High'
        elif med_min <= count <= med_max: return 'Medium'
        elif count <= low_thresh: return 'Low'
        else: return 'Discard' # The Gap Zone
        
    cluster_risk = cluster_counts.apply(assign_risk)
    df['Risk_Zone'] = df['Cluster'].map(cluster_risk)
    
    # Remove nodes in the gap
    df = df[df['Risk_Zone'] != 'Discard'].copy()
    
    print(f"Risk Zone Distribution (Filtered with Gaps):\n{df['Risk_Zone'].value_counts().to_string()}")
    
    # 2. Advanced Feature Engineering
    print("Engineering interaction features (Urban_Speed, IsRushHour, Hour_Sin/Cos, DayNight_Context, Junction_Complexity)...")
    
    if 'Time' in df.columns:
        df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce').dt.hour
        df['Hour'] = df['Hour'].fillna(df['Hour'].median())
        df['Hour'] = df['Hour'].round().clip(0, 23).astype(int)
        df['IsNight'] = ((df['Hour'] < 6) | (df['Hour'] >= 20)).astype(int)
        # FEATURE: Rush Hour Flag
        df['IsRushHour'] = df['Hour'].apply(lambda x: 1 if x in [7,8,9,17,18,19] else 0)
        
        # FEATURE: Temporal Sine/Cosine Transformation ("Clock-face" feature)
        df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24.0)
        df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24.0)
        
        # FEATURE: Interaction Grid "The Friday Night Factor"
        if 'Day_of_Week' in df.columns:
            df['DayNight_Context'] = df['Day_of_Week'].astype(str) + "_" + df['IsNight'].astype(str)
        
    # FEATURE: Urban Speed Intensity
    if 'Urban_or_Rural_Area' in df.columns and 'Speed_limit' in df.columns:
        df['Urban_Speed_Net'] = df['Urban_or_Rural_Area'] * df['Speed_limit']
        
    # FEATURE: Junction complexity
    if 'Junction_Detail' in df.columns and 'Junction_Control' in df.columns:
        df['Junction_Complexity'] = df['Junction_Detail'] + df['Junction_Control']

    # Save a small sample of coordinates for Visualization later
    sample_df = df[['Latitude', 'Longitude', 'Risk_Zone']].copy()
    # Mask out the "Discard" nodes (already filtered, but for safety)
    sample_df = sample_df[sample_df['Risk_Zone'] != 'Discard']
    sample_df.sample(n=min(50000, len(sample_df)), random_state=42).to_csv(
        os.path.join(output_dir, '6_spatial_sample.csv'), index=False
    )
    
    # We deliberately drop geography so the ML models MUST learn the pure environmental signature!
    drop_cols = ['Latitude', 'Longitude', 'Local_Authority_(District)', 'Police_Force', 
                 'Accident_Severity', 'Cluster', 'Number_of_Casualties', 'Number_of_Vehicles', 'Time']
        
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    df.dropna(subset=['Risk_Zone'], inplace=True)

    # Once geography is removed, many raw accidents collapse into the exact same
    # supervised-learning pattern. Keeping repeated copies makes the later
    # holdout split look easier than it really is, so prune exact duplicates
    # before balancing and model training.
    supervised_key_cols = [col for col in df.columns]
    duplicate_rows = int(df.duplicated(subset=supervised_key_cols).sum())
    if duplicate_rows > 0:
        print(f"Removing {duplicate_rows} exact duplicate feature+target rows before balancing.")
        df = df.drop_duplicates(subset=supervised_key_cols).copy()
    
    # 3. Balancing for Supervised ML Training
    print("Balancing dataset across Risk Zones for Supervised ML classifiers...")
    sampled_dfs = []
    target_samples = 40000 # Keeping dataset lightweight and fully balanced for class fairness
    
    for zone in ['Low', 'Medium', 'High']:
        zone_df = df[df['Risk_Zone'] == zone]
        if len(zone_df) > target_samples:
            sampled_dfs.append(zone_df.sample(n=target_samples, random_state=42))
        else:
            # Over-sample nicely if a bit short, which shouldn't happen usually here
            sampled_dfs.append(zone_df.sample(n=target_samples, random_state=42, replace=True))
            
    df_balanced = pd.concat(sampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Balanced Dataset Shape: {df_balanced.shape}")
    
    # 4. Encoding
    y = df_balanced['Risk_Zone'].map({'Low': 0, 'Medium': 1, 'High': 2})
    X = df_balanced.drop(columns=['Risk_Zone'])

    # Save a correlation matrix over the interpretable numeric / engineered
    # feature subset before supervised training. We avoid arbitrary label-encoded
    # nominal categories here because their Pearson correlation is misleading.
    corr_cols = [col for col in CORRELATION_FEATURE_COLS if col in X.columns]
    if corr_cols:
        corr_source = X[corr_cols].copy()
        corr_source['Risk_Zone_Code'] = y
        corr_matrix = corr_source.corr(numeric_only=True)
        corr_matrix.to_csv(os.path.join(output_dir, '6_feature_correlation_matrix.csv'))
        print(f"Saved feature correlation matrix with columns: {corr_matrix.columns.tolist()}")
    
    encoders = {}
    for col in X.columns:
        if X[col].dtype == object or str(X[col].dtype) in ['category', 'string']:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
            
    joblib.dump(encoders, os.path.join(output_dir, '6_label_encoders.joblib'))
    
    X.to_csv(os.path.join(output_dir, '6_X_final.csv'), index=False)
    y.to_csv(os.path.join(output_dir, '6_y_final.csv'), index=False)
    print("Saved ready-for-ML arrays inside output/")

if __name__ == "__main__":
    run()
