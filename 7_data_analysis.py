import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

def run():
    print("\n--- STEP 7: Machine Learning (Multiple Experts Upgrade) ---")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'output')
    x_path = os.path.join(output_dir, '6_X_final.csv')
    y_path = os.path.join(output_dir, '6_y_final.csv')
    
    if not os.path.exists(x_path):
        print(f"Error: {x_path} not found.")
        return
        
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path).iloc[:, 0] 
    
    print("Splitting train/test data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- [Expert 1: Unsupervised KMeans covered in Step 6] ---

    print("\n[Expert 2] Training Random Forest Classifier...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
    
    print("\n[Expert 3] Training XGBoost Classifier...")
    xgb_model = xgb.XGBClassifier(n_estimators=150, max_depth=10, learning_rate=0.1, n_jobs=-1, random_state=42, eval_metric='mlogloss')
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    print(f"XGBoost Accuracy: {accuracy_score(y_test, xgb_preds):.4f}")

    print("\n[Expert 4] Training LightGBM Classifier (High-Accuracy Expert)...")
    lgbm_model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=127,       # More leaves = more granular splits
        max_depth=12,
        min_child_samples=10,
        subsample=0.8,        # Row subsampling for regularization
        colsample_bytree=0.8, # Feature subsampling
        reg_alpha=0.1,
        reg_lambda=0.2,
        n_jobs=-1,
        random_state=42,
        importance_type='gain',
        verbose=-1
    )
    lgbm_model.fit(X_train, y_train)
    lgbm_preds = lgbm_model.predict(X_test)
    print(f"LightGBM Accuracy: {accuracy_score(y_test, lgbm_preds):.4f}")
    
    print("\n--- Triple Expert Comparison & Final Evaluation ---")
    target_names = ['Low Risk', 'Medium Risk', 'High Risk']
    
    for name, preds in zip(['Random Forest', 'XGBoost', 'LightGBM'], [rf_preds, xgb_preds, lgbm_preds]):
        print(f"\n{name} Report:")
        print(classification_report(y_test, preds, target_names=target_names))
    
    # Save all predictions for multi-panel visualization in Step 8
    preds_df = pd.DataFrame({
        'y_test': y_test, 
        'y_pred_rf': rf_preds, 
        'y_pred_xgb': xgb_preds,
        'y_pred_lgbm': lgbm_preds
    })
    preds_df.to_csv(os.path.join(output_dir, '7_predictions.csv'), index=False)
    
    # Feature Importances from the best model (typically LightGBM)
    fi = pd.DataFrame({
        'Feature': X.columns,
        'Importance': lgbm_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\nTop 5 Predictive Features (LightGBM):")
    print(fi.head(5).to_string(index=False))
    
    fi.to_csv(os.path.join(output_dir, '7_feature_importances.csv'), index=False)
    joblib.dump(lgbm_model, os.path.join(output_dir, '7_best_model.joblib'))
    
    print(f"All experts evaluated and saved successfully inside {output_dir}.")

if __name__ == "__main__":
    run()
