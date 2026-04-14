import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

def run():
    print("\n--- STEP 7: Machine Learning (Multiple Experts) ---")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'output')
    x_path = os.path.join(output_dir, '6_X_final.csv')
    y_path = os.path.join(output_dir, '6_y_final.csv')
    
    if not os.path.exists(x_path):
        print(f"Error: {x_path} not found.")
        return
        
    X = pd.read_csv(x_path)
    # y was saved as a single column without header or with a header 'Risk_Zone'.
    # We read the first column to be safe.
    y = pd.read_csv(y_path).iloc[:, 0] 
    
    print("Splitting train/test data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\n[Expert 2] Training Random Forest Classifier...")
    # Expert 1 was the Unsupervised Clustering in Step 6
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    
    print("\n[Expert 3] Training XGBoost Classifier...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=150, 
        max_depth=10, 
        learning_rate=0.1, 
        n_jobs=-1, 
        random_state=42, 
        eval_metric='mlogloss'
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_preds)
    print(f"XGBoost Accuracy: {xgb_acc:.4f}")
    
    print("\n--- Model Comparison & Evaluation ---")
    target_names = ['Low Risk', 'Medium Risk', 'High Risk']
    
    print("Random Forest Classification Report:")
    print(classification_report(y_test, rf_preds, target_names=target_names))
    
    print("XGBoost Classification Report:")
    print(classification_report(y_test, xgb_preds, target_names=target_names))
    
    # Save predictions from both models for visualization in step 8
    preds_df = pd.DataFrame({'y_test': y_test, 'y_pred_rf': rf_preds, 'y_pred_xgb': xgb_preds})
    preds_df.to_csv(os.path.join(output_dir, '7_predictions.csv'), index=False)
    
    # Feature Importances from XGBoost
    fi = pd.DataFrame({
        'Feature': X.columns,
        'Importance': xgb_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\nTop 5 Environmental Predictors of High-Risk Zones:")
    print(fi.head(5).to_string(index=False))
    
    fi.to_csv(os.path.join(output_dir, '7_feature_importances.csv'), index=False)
    joblib.dump(xgb_model, os.path.join(output_dir, '7_best_model.joblib'))
    
    print("Models and evaluation metrics saved successfully inside output/.")

if __name__ == "__main__":
    run()
