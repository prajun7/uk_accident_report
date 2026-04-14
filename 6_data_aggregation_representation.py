import os
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import LabelEncoder
import joblib

def run():
    print("\n--- STEP 6: Risk Zone Clustering & Aggregation ---")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'output')
    in_path = os.path.join(output_dir, '5_cleansed_data.csv')
    
    if not os.path.exists(in_path):
        print(f"Error: {in_path} not found.")
        return
        
    df = pd.read_csv(in_path, low_memory=False)
    
    # 1. Unsupervised Learning: Spatial Hotspot Clustering
    print("Performing spatial clustering (Unsupervised Learning) to define Risk Zones...")
    coords = df[['Latitude', 'Longitude']].dropna()
    
    # Use MiniBatchKMeans to handle 1.7M rows quickly
    kmeans = MiniBatchKMeans(n_clusters=250, random_state=42, batch_size=10000, n_init='auto')
    df.loc[coords.index, 'Cluster'] = kmeans.fit_predict(coords)
    
    # Calculate density (accidents per cluster)
    cluster_counts = df['Cluster'].value_counts()
    
    # Define Risk Zones based on accident density per cluster
    # High Risk = Top 15% of clusters by accident volume
    high_thresh = cluster_counts.quantile(0.85)
    med_thresh = cluster_counts.quantile(0.50)
    
    def assign_risk(count):
        if count >= high_thresh: return 'High'
        elif count >= med_thresh: return 'Medium'
        else: return 'Low'
        
    cluster_risk = cluster_counts.apply(assign_risk)
    df['Risk_Zone'] = df['Cluster'].map(cluster_risk)
    
    print(f"Risk Zone Distribution (Raw):\n{df['Risk_Zone'].value_counts().to_string()}")
    
    # Save a small sample of coordinates for Visualization later
    sample_df = df[['Latitude', 'Longitude', 'Risk_Zone']].copy()
    sample_df['Risk_Zone'] = sample_df['Risk_Zone'].fillna('Unknown') # Safe drop
    sample_df = sample_df[sample_df['Risk_Zone'] != 'Unknown']
    sample_df.sample(n=min(50000, len(sample_df)), random_state=42).to_csv(
        os.path.join(output_dir, '6_spatial_sample.csv'), index=False
    )
    
    # 2. Feature Engineering (Environmental & Infrastructural)
    # We deliberately drop geography so the ML models MUST learn the pure environmental signature!
    drop_cols = ['Latitude', 'Longitude', 'Local_Authority_(District)', 'Police_Force', 
                 'Accident_Severity', 'Cluster', 'Number_of_Casualties', 'Number_of_Vehicles']
    
    if 'Time' in df.columns:
        df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce').dt.hour
        df['Hour'] = df['Hour'].fillna(df['Hour'].median())
        df['IsNight'] = ((df['Hour'] < 6) | (df['Hour'] >= 20)).astype(int)
        drop_cols.append('Time')
        
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    df.dropna(subset=['Risk_Zone'], inplace=True)
    
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
    
    encoders = {}
    for col in X.columns:
        if X[col].dtype == object or str(X[col].dtype) == 'category':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
            
    joblib.dump(encoders, os.path.join(output_dir, '6_label_encoders.joblib'))
    
    X.to_csv(os.path.join(output_dir, '6_X_final.csv'), index=False)
    y.to_csv(os.path.join(output_dir, '6_y_final.csv'), index=False)
    print("Saved ready-for-ML arrays inside output/")

if __name__ == "__main__":
    run()
