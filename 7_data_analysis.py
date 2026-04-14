import os

import joblib
import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from category_encoders import TargetEncoder
from category_encoders.wrapper import PolynomialWrapper
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

ENGINEERED_FEATURE_FORMULAS = {
    "IsNight": "1 if Hour < 6 or Hour >= 20 else 0",
    "IsRushHour": "1 if Hour in {7,8,9,17,18,19} else 0",
    "Hour_Sin": "sin(2*pi*Hour/24)",
    "Hour_Cos": "cos(2*pi*Hour/24)",
    "DayNight_Context": "str(Day_of_Week) + '_' + str(IsNight)",
    "Urban_Speed_Net": "Urban_or_Rural_Area * Speed_limit",
    "Junction_Complexity": "Junction_Detail + Junction_Control",
    "1st_Road_Number_0": "Target-encoded 1st_Road_Number (class 0, out-of-fold)",
    "1st_Road_Number_1": "Target-encoded 1st_Road_Number (class 1, out-of-fold)",
    "2nd_Road_Number_0": "Target-encoded 2nd_Road_Number (class 0, out-of-fold)",
    "2nd_Road_Number_1": "Target-encoded 2nd_Road_Number (class 1, out-of-fold)",
    "DayNight_Context_0": "Target-encoded DayNight_Context (class 0, out-of-fold)",
    "DayNight_Context_1": "Target-encoded DayNight_Context (class 1, out-of-fold)",
}


def apply_oof_target_encoding(X_train, X_test, y_train, target_cols, n_splits=5):
    """
    Use out-of-fold target encoding on the training split so each row is encoded
    only from other folds, then fit a final encoder on the full train split for test data.
    """
    if not target_cols:
        return X_train.copy(), X_test.copy(), None

    X_train_for_encoding = X_train.copy()
    X_test_for_encoding = X_test.copy()

    for col in target_cols:
        X_train_for_encoding[col] = X_train_for_encoding[col].astype("string")
        X_test_for_encoding[col] = X_test_for_encoding[col].astype("string")

    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    encoded_folds = []

    for fit_idx, valid_idx in folds.split(X_train_for_encoding, y_train):
        fold_encoder = PolynomialWrapper(
            TargetEncoder(cols=target_cols, min_samples_leaf=20, smoothing=10.0)
        )
        fold_encoder.fit(X_train_for_encoding.iloc[fit_idx], y_train.iloc[fit_idx])

        X_valid_encoded = fold_encoder.transform(X_train_for_encoding.iloc[valid_idx])
        X_valid_encoded.index = X_train.index[valid_idx]
        encoded_folds.append(X_valid_encoded)

    X_train_encoded = pd.concat(encoded_folds).sort_index()

    final_encoder = PolynomialWrapper(
        TargetEncoder(cols=target_cols, min_samples_leaf=20, smoothing=10.0)
    )
    final_encoder.fit(X_train_for_encoding, y_train)
    X_test_encoded = final_encoder.transform(X_test_for_encoding)

    X_train_encoded = X_train_encoded.reindex(columns=X_test_encoded.columns)
    return X_train_encoded, X_test_encoded, final_encoder


def ensure_numeric_features(X_train, X_test):
    """
    Safety net in case any string/categorical columns survive Step 6.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    for col in X_train.columns:
        dtype_name = str(X_train[col].dtype)
        if X_train[col].dtype == object or dtype_name in {"category", "string"}:
            encoder = LabelEncoder()
            encoder.fit(pd.concat([X_train[col], X_test[col]], axis=0).astype(str))
            X_train[col] = encoder.transform(X_train[col].astype(str))
            X_test[col] = encoder.transform(X_test[col].astype(str))

    return X_train, X_test


def scale_for_neural_network(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def save_feature_importance(output_dir, model_name, feature_names, importances, top_n=5):
    fi = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances,
        'Model': model_name,
    })
    fi['Is_Engineered'] = fi['Feature'].isin(ENGINEERED_FEATURE_FORMULAS)
    fi['Formula'] = fi['Feature'].map(ENGINEERED_FEATURE_FORMULAS).fillna('')
    fi = fi.sort_values(by='Importance', ascending=False)
    safe_name = model_name.lower().replace(" ", "_")
    fi.to_csv(os.path.join(output_dir, f'7_feature_importances_{safe_name}.csv'), index=False)
    print(f"\nTop {top_n} Predictive Features ({model_name}):")
    preview = fi.head(top_n).copy()
    preview['Feature'] = preview.apply(
        lambda row: f"{row['Feature']} *" if row['Is_Engineered'] else row['Feature'],
        axis=1
    )
    print(preview[['Feature', 'Importance']].to_string(index=False))
    return fi


def build_metric_row(model_name, y_true, y_pred):
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average='macro',
        zero_division=0,
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average='weighted',
        zero_division=0,
    )
    return {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Macro_Precision': macro_precision,
        'Macro_Recall': macro_recall,
        'Macro_F1': macro_f1,
        'Weighted_F1': weighted_f1,
    }


def count_exact_feature_overlap(X_train, X_test):
    train_hashes = set(pd.util.hash_pandas_object(X_train, index=False))
    test_hashes = pd.util.hash_pandas_object(X_test, index=False)
    return int(test_hashes.isin(train_hashes).sum())

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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    overlap_rows = count_exact_feature_overlap(X_train, X_test)
    print(f"Exact train/test feature-row overlap: {overlap_rows} of {len(X_test)} test rows")

    print("Applying leakage-free out-of-fold target encoding on high-cardinality features...")
    candidate_target_cols = [
        "1st_Road_Number",
        "2nd_Road_Number",
        "Local_Authority_(District)",
        "DayNight_Context",
    ]
    target_cols = [
        col for col in candidate_target_cols
        if col in X_train.columns and X_train[col].nunique(dropna=False) > 10
    ]

    if target_cols:
        print(f"Encoding columns: {', '.join(target_cols)}")
        X_train, X_test, target_encoder = apply_oof_target_encoding(
            X_train, X_test, y_train, target_cols
        )
    else:
        target_encoder = None

    X_train, X_test = ensure_numeric_features(X_train, X_test)
    print(f"Features ready. Final shape after encoding: {X_train.shape}")

    # --- [Expert 1: Unsupervised KMeans covered in Step 6] ---

    print("\n[Expert 2] Training Random Forest Classifier...")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=20, n_jobs=-1, random_state=RANDOM_STATE
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
    
    print("\n[Expert 3] Training XGBoost Classifier...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=10,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
        objective="multi:softprob",
        num_class=3,
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    print(f"XGBoost Accuracy: {accuracy_score(y_test, xgb_preds):.4f}")

    print("\n[Expert 4] Training LightGBM Classifier...")
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
        random_state=RANDOM_STATE,
        objective="multiclass",
        num_class=3,
        importance_type='gain',
        verbose=-1
    )
    lgbm_model.fit(X_train, y_train)
    lgbm_preds = lgbm_model.predict(X_test)
    print(f"LightGBM Accuracy: {accuracy_score(y_test, lgbm_preds):.4f}")

    # The earlier stacking ensemble was removed on purpose: it added noticeable
    # training/inference overhead while giving only a marginal holdout lift.
    # A compact MLP gives us a non-tree baseline to compare against instead.
    print("\n[Expert 5] Training Neural Network Classifier...")
    X_train_nn, X_test_nn, nn_scaler = scale_for_neural_network(X_train, X_test)
    nn_model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        batch_size=512,
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=RANDOM_STATE,
        verbose=False,
    )
    nn_model.fit(X_train_nn, y_train)
    nn_preds = nn_model.predict(X_test_nn)
    print(f"Neural Network Accuracy: {accuracy_score(y_test, nn_preds):.4f}")
    
    print("\n--- Model Comparison & Final Evaluation ---")
    target_names = ['Low Risk', 'Medium Risk', 'High Risk']
    metrics_rows = [
        build_metric_row('Random Forest', y_test, rf_preds),
        build_metric_row('XGBoost', y_test, xgb_preds),
        build_metric_row('LightGBM', y_test, lgbm_preds),
        build_metric_row('Neural Network', y_test, nn_preds),
    ]
    model_scores = {row['Model']: row['Accuracy'] for row in metrics_rows}
    
    for name, preds in zip(['Random Forest', 'XGBoost', 'LightGBM', 'Neural Network'], 
                           [rf_preds, xgb_preds, lgbm_preds, nn_preds]):
        print(f"\n{name} Report:")
        print(classification_report(y_test, preds, target_names=target_names))

    scores_df = pd.DataFrame(metrics_rows).sort_values(by="Accuracy", ascending=False)
    scores_df.to_csv(os.path.join(output_dir, '7_model_scores.csv'), index=False)
    print("\nCompact score summary:")
    print(scores_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    best_model_name = scores_df.iloc[0]["Model"]
    best_model = {
        'Random Forest': rf,
        'XGBoost': xgb_model,
        'LightGBM': lgbm_model,
        'Neural Network': nn_model,
    }[best_model_name]
    print(f"\nBest model on the holdout split: {best_model_name} ({scores_df.iloc[0]['Accuracy']:.4f})")
    
    # Save all predictions for multi-panel visualization in Step 8
    preds_df = pd.DataFrame({
        'y_test': y_test, 
        'y_pred_rf': rf_preds, 
        'y_pred_xgb': xgb_preds,
        'y_pred_lgbm': lgbm_preds,
        'y_pred_nn': nn_preds,
    })
    preds_df.to_csv(os.path.join(output_dir, '7_predictions.csv'), index=False)
    
    rf_fi = save_feature_importance(
        output_dir,
        'Random Forest',
        X_train.columns,
        rf.feature_importances_,
    )
    xgb_fi = save_feature_importance(
        output_dir,
        'XGBoost',
        X_train.columns,
        xgb_model.feature_importances_,
    )
    lgbm_fi = save_feature_importance(
        output_dir,
        'LightGBM',
        X_train.columns,
        lgbm_model.feature_importances_,
    )

    pd.concat([rf_fi, xgb_fi, lgbm_fi], ignore_index=True).to_csv(
        os.path.join(output_dir, '7_feature_importances_all_models.csv'), index=False
    )
    joblib.dump(best_model, os.path.join(output_dir, '7_best_model.joblib'))
    joblib.dump(
        {
            'model_name': best_model_name,
            'model': best_model,
            'feature_columns': X_train.columns.tolist(),
            'raw_feature_columns': X.columns.tolist(),
            'target_encoder': target_encoder,
            'target_encoder_cols': target_cols,
            'target_names': target_names,
            'nn_scaler': nn_scaler if best_model_name == 'Neural Network' else None,
        },
        os.path.join(output_dir, '7_model_bundle.joblib'),
    )
    
    print(f"All experts evaluated and saved successfully inside {output_dir}.")

if __name__ == "__main__":
    run()
