import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def run():
    print("\n--- STEP 8: Data Visualization ---")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'output')
    fi_path = os.path.join(output_dir, '7_feature_importances.csv')
    
    if not os.path.exists(fi_path):
        print(f"Error: Could not find {fi_path}. Run Step 7 first.")
        return

    # 1. Feature Importance Plot
    print("Generating Feature Importance Bar Chart...")
    fi_df = pd.read_csv(fi_path)
    # Get top 8 features
    top_fi = fi_df.head(8)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=top_fi, palette='viridis')
    plt.title('Top Predictive Environmental Factors For Accident Severity')
    plt.xlabel('Relative Importance (Random Forest)')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    out_img = os.path.join(output_dir, '8_feature_importances.png')
    plt.savefig(out_img)
    plt.close()
    
    print(f"Saved Visualization to {out_img}")

if __name__ == "__main__":
    run()
