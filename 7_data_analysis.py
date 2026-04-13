import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def run():
    print("\n--- STEP 7: Data Analysis ---")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'output')
    x_path = os.path.join(output_dir, '6_X_final.csv')
    y_path = os.path.join(output_dir, '6_y_final.csv')
    
    if not os.path.exists(x_path):
        print(f"Error: Could not find {x_path}.")
        return

    X = pd.read_csv(x_path, low_memory=False)
    y = pd.read_csv(y_path, low_memory=False)['Accident_Severity']
    
    # Due to extreme size, we gracefully sample to avoid memory crash on non-cluster machines during development
    # If the user has a massive cluster, they can comment this out.
    if len(X) > 250000:
        print(f"Dataset is {len(X)} rows. Subsampling strictly for local memory constraints to 500k.")
        sample_idx = X.sample(n=500000, random_state=42).index
        X = X.loc[sample_idx]
        y = y.loc[sample_idx]

    print("Splitting train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Classifier to predict Accident Severity...")
    # Accident Severity ranges typically from 1(Fatal) to 3(Slight), softly weigh minority
    rf = RandomForestClassifier(n_estimators=100, max_depth=25, n_jobs=-1, random_state=42, class_weight={1: 5, 2: 2, 3: 1})
    rf.fit(X_train, y_train)
    
    print("Evaluating Model...")
    y_pred = rf.predict(X_test)
    
    # Save the report for Visualization step
    report = classification_report(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\nAccuracy: {acc:.4f}")
    print("Classification Report:")
    print(report)
    
    # Determine the most predictive environmental variables
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\nTop 5 Predictive Features:")
    print(feature_importances.head(5))
    
    fi_path = os.path.join(output_dir, '7_feature_importances.csv')
    feature_importances.to_csv(fi_path, index=False)
    
    model_path = os.path.join(output_dir, '7_rf_model.joblib')
    joblib.dump(rf, model_path)
    
    # Save predictions for Confusion Matrix visual in Step 8
    preds_df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
    preds_df.to_csv(os.path.join(output_dir, '7_predictions.csv'), index=False)
    
    print(f"Saved feature importances, predictions, and trained model inside {output_dir}.")

if __name__ == "__main__":
    run()
