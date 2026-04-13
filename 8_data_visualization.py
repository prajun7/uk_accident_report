import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def run():
    print("\n--- STEP 8: Lifecycle Data Visualization ---")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'output')
    
    x_path = os.path.join(output_dir, '6_X_final.csv')
    y_path = os.path.join(output_dir, '6_y_final.csv')
    preds_path = os.path.join(output_dir, '7_predictions.csv')
    
    if not os.path.exists(preds_path):
        print(f"Error: Could not find {preds_path}. Run Step 7 first to generate ML predictions.")
        return

    # Load data
    print("Loading aggregated CSVs for visual distributions...")
    X = pd.read_csv(x_path, low_memory=False)
    y = pd.read_csv(y_path, low_memory=False)['Accident_Severity']
    preds_df = pd.read_csv(preds_path)

    # We will use Seborn plotting themes for a premium look
    sns.set_theme(style="whitegrid")
    
    # -------------------------------------------------------------
    # 1. THE IMBALANCED LANDSCAPE (Target Distribution)
    # -------------------------------------------------------------
    print("Generating Chart 1: Imbalanced Target Landscape")
    plt.figure(figsize=(8, 8))
    # 1=Fatal, 2=Serious, 3=Slight -> map to text
    y_labels = y.map({1: 'Fatal', 2: 'Serious', 3: 'Slight'})
    y_counts = y_labels.value_counts(normalize=True) * 100
    
    plt.pie(y_counts, labels=y_counts.index, autopct='%1.1f%%', 
            colors=['steelblue', 'orange', 'crimson'], startangle=140, explode=[0.05]*len(y_counts))
    plt.title('Stage: Pre-Processing\nThe Real-World Traffic Accident Imbalance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '8_1_target_imbalance.png'))
    plt.close()

    # -------------------------------------------------------------
    # 2. FEATURE CORRELATION HEATMAP
    # -------------------------------------------------------------
    print("Generating Chart 2: Feature Correlation Heatmap")
    plt.figure(figsize=(12, 10))
    corr = X.corr(method='spearman')
    # Custom diverging palette
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, cbar_kws={"shrink": .8}, linewidths=.5)
    plt.title('Stage: Aggregation\nEngineered Feature Correlation Map')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '8_2_correlation_heatmap.png'))
    plt.close()

    # -------------------------------------------------------------
    # 3. TEMPORAL CRASH FREQUENCIES
    # -------------------------------------------------------------
    print("Generating Chart 3: Temporal Exploratory Data Analysis")
    if 'Hour' in X.columns and 'IsNight' in X.columns:
        plt.figure(figsize=(12, 6))
        # Overlay KDE for day vs night explicitly to see rush hour shifts
        sns.kdeplot(data=X, x='Hour', hue='IsNight', fill=True, common_norm=False, palette=['darkorange', 'midnightblue'])
        plt.title('Stage: Feature Engineering\nAccident Density Spikes by Hour of Day/Night')
        plt.xlabel('Hour (0-23)')
        plt.ylabel('Crash Density')
        plt.xlim(0, 23)
        plt.xticks(np.arange(0, 24, 2))
        # Replace 0, 1 with Day, Night
        plt.legend(title='Condition', labels=['Night (8pm-6am)', 'Daytime'])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '8_3_temporal_density.png'))
        plt.close()
        
    # -------------------------------------------------------------
    # 4. MACHINE LEARNING CONFUSION MATRIX
    # -------------------------------------------------------------
    print("Generating Chart 4: Machine Learning Predictor Heatmap")
    plt.figure(figsize=(9, 7))
    cm = confusion_matrix(preds_df['y_test'], preds_df['y_pred'], labels=[1, 2, 3])
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, 
                xticklabels=['Predict: Fatal', 'Predict: Serious', 'Predict: Slight'],
                yticklabels=['Actual: Fatal', 'Actual: Serious', 'Actual: Slight'])
    
    plt.title('Stage: ML Evaluation\nRandom Forest Confusion Matrix Validation')
    plt.ylabel('True Real-World Class')
    plt.xlabel('Model Predicted Class')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '8_4_ml_confusion_matrix.png'))
    plt.close()

    print("Successfully generated all 4 analytical milestone figures inside the output/ folder!")

if __name__ == "__main__":
    run()
