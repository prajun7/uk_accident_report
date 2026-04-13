import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def run():
    print("\n--- STEP 6: Data Aggregation & Representation ---")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'output')
    in_path = os.path.join(output_dir, '5_cleansed_data.csv')
    
    if not os.path.exists(in_path):
        print(f"Error: Could not find {in_path}.")
        return

    df = pd.read_csv(in_path, low_memory=False)
    
    # 1. Feature Engineering
    print("Extracting time-based features (Hour, IsNight).")
    if 'Time' in df.columns:
        # Assuming format HH:MM
        df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce').dt.hour
        # Fill missing hours with median
        df['Hour'] = df['Hour'].fillna(df['Hour'].median())
        df['IsNight'] = ((df['Hour'] < 6) | (df['Hour'] >= 20)).astype(int)
        df.drop(columns=['Time'], inplace=True)
        
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        df['Month'] = df['Date'].dt.month.fillna(6).astype(int)
        df.drop(columns=['Date'], inplace=True)
        
    # 2. Label Encoding for remaining categoricals
    print("Label encoding categorical variables.")
    encoders = {}
    for col in df.columns:
        if df[col].dtype == object:
            le = LabelEncoder()
            # We handle unknown later in inference, so we just fit on known string
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            
    # Save the encoders for inference
    joblib.dump(encoders, os.path.join(output_dir, '6_label_encoders.joblib'))
    print("Saved label encoders to 6_label_encoders.joblib")
            
    # 3. Separate Matrix
    y = df['Accident_Severity']
    X = df.drop(columns=['Accident_Severity'])
    print(f"Generated X shape: {X.shape}, y shape: {y.shape}")
    
    out_x = os.path.join(output_dir, '6_X_final.csv')
    out_y = os.path.join(output_dir, '6_y_final.csv')
    
    X.to_csv(out_x, index=False)
    y.to_csv(out_y, index=False)
    
    print("Saved ready-for-ML arrays 6_X_final.csv and 6_y_final.csv.")

if __name__ == "__main__":
    run()
