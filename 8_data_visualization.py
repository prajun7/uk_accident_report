import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def run():
    print("\n--- STEP 8: Final Project Visualizations ---")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'output')
    
    spatial_path = os.path.join(output_dir, '6_spatial_sample.csv')
    preds_path = os.path.join(output_dir, '7_predictions.csv')
    fi_path = os.path.join(output_dir, '7_feature_importances.csv')
    
    sns.set_theme(style="whitegrid")
    
    # 1. Unsupervised Learning Map (DBSCAN/KMeans Clusters)
    if os.path.exists(spatial_path):
        print("Generating Chart 1: Geospatial Risk Clusters (Unsupervised Learning)")
        spatial_df = pd.read_csv(spatial_path)
        plt.figure(figsize=(8, 10))
        # Custom color map for severity
        colors = {'Low': 'mediumseagreen', 'Medium': 'gold', 'High': 'crimson'}
        for zone in ['Low', 'Medium', 'High']:
            subset = spatial_df[spatial_df['Risk_Zone'] == zone]
            plt.scatter(subset['Longitude'], subset['Latitude'], s=2, alpha=0.5, c=colors[zone], label=f'{zone} Risk')
        
        plt.title('Unsupervised Geospatial Risk Clustering\n(UK Road Network Map)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend(markerscale=5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '8_1_spatial_risk_map.png'), dpi=150)
        plt.close()
        
    # 2. Confusion Matrices Comparison
    if os.path.exists(preds_path):
        print("Generating Chart 2: ML Expert Comparison (Confusion Matrices)")
        preds = pd.read_csv(preds_path)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        cm_rf = confusion_matrix(preds['y_test'], preds['y_pred_rf'], labels=[0, 1, 2])
        sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                    xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
        axes[0].set_title('Expert 2: Random Forest Classifier')
        axes[0].set_ylabel('True Risk Zone')
        axes[0].set_xlabel('Predicted Risk Zone')
        
        cm_xgb = confusion_matrix(preds['y_test'], preds['y_pred_xgb'], labels=[0, 1, 2])
        sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Oranges", ax=axes[1],
                    xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
        axes[1].set_title('Expert 3: XGBoost Classifier')
        axes[1].set_xlabel('Predicted Risk Zone')
        
        plt.suptitle("Supervised Classification Experts Evaluation")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '8_2_model_comparison_cm.png'))
        plt.close()
        
    # 3. Feature Importance
    if os.path.exists(fi_path):
        print("Generating Chart 3: Top Environmental Risk Predictors")
        fi = pd.read_csv(fi_path).head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=fi, x='Importance', y='Feature', palette='viridis', hue='Feature', legend=False)
        plt.title('Top 10 Environmental Predictors of High-Risk Zones\n(XGBoost Feature Importance)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '8_3_feature_importance.png'))
        plt.close()
        
    print("All visualizations successfully generated in output/ folder!")

if __name__ == "__main__":
    run()
