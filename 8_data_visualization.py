import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

ENGINEERED_FEATURE_FORMULAS = {
    "IsNight": "1 if Hour < 6 or Hour >= 20 else 0",
    "IsRushHour": "1 if Hour in {7,8,9,17,18,19} else 0",
    "Hour_Sin": "sin(2*pi*Hour/24)",
    "Hour_Cos": "cos(2*pi*Hour/24)",
    "Urban_Speed_Net": "Urban_or_Rural_Area * Speed_limit",
    "Junction_Complexity": "Junction_Detail + Junction_Control",
}

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


def format_feature_label(row):
    return f"{row['Feature']} *" if bool(row.get('Is_Engineered', False)) else row['Feature']


def build_formula_note(fi_df):
    engineered = fi_df[fi_df['Is_Engineered'] == True][['Feature', 'Formula']].drop_duplicates()
    if engineered.empty:
        return ""
    lines = [f"* {feature} = {formula}" for feature, formula in engineered.itertuples(index=False, name=None)]
    return "\n".join(lines)


def mark_engineered_label(name):
    return f"{name} *" if name in ENGINEERED_FEATURE_FORMULAS else name


def build_formula_note_from_labels(labels):
    lines = [
        f"* {label} = {ENGINEERED_FEATURE_FORMULAS[label]}"
        for label in labels
        if label in ENGINEERED_FEATURE_FORMULAS
    ]
    return "\n".join(lines)


def build_metrics_table(preds_df):
    model_cols = [
        ('Random Forest', 'y_pred_rf'),
        ('XGBoost', 'y_pred_xgb'),
        ('LightGBM', 'y_pred_lgbm'),
        ('Neural Network', 'y_pred_nn'),
    ]
    rows = []
    for model_name, pred_col in model_cols:
        if pred_col not in preds_df.columns:
            continue
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            preds_df['y_test'],
            preds_df[pred_col],
            average='macro',
            zero_division=0,
        )
        _, _, weighted_f1, _ = precision_recall_fscore_support(
            preds_df['y_test'],
            preds_df[pred_col],
            average='weighted',
            zero_division=0,
        )
        rows.append({
            'Model': model_name,
            'Accuracy': accuracy_score(preds_df['y_test'], preds_df[pred_col]),
            'Macro_Precision': macro_precision,
            'Macro_Recall': macro_recall,
            'Macro_F1': macro_f1,
            'Weighted_F1': weighted_f1,
        })
    return pd.DataFrame(rows)

def run():
    print("\n--- STEP 8: Final Project Visualizations ---")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'output')
    
    spatial_path = os.path.join(output_dir, '6_spatial_sample.csv')
    x_path = os.path.join(output_dir, '6_X_final.csv')
    y_path = os.path.join(output_dir, '6_y_final.csv')
    corr_path = os.path.join(output_dir, '6_feature_correlation_matrix.csv')
    preds_path = os.path.join(output_dir, '7_predictions.csv')
    scores_path = os.path.join(output_dir, '7_model_scores.csv')
    fi_all_models_path = os.path.join(output_dir, '7_feature_importances_all_models.csv')
    legacy_feature_compare_path = os.path.join(output_dir, '8_4_feature_importance_comparison.png')
    
    sns.set_theme(style="whitegrid")

    if os.path.exists(legacy_feature_compare_path):
        os.remove(legacy_feature_compare_path)
        print("Removed legacy duplicate feature-comparison chart: 8_4_feature_importance_comparison.png")

    if not os.path.exists(corr_path) and os.path.exists(x_path) and os.path.exists(y_path):
        print("Building fallback correlation matrix from Step 6 outputs...")
        X = pd.read_csv(x_path)
        y = pd.read_csv(y_path).iloc[:, 0]
        corr_cols = [col for col in CORRELATION_FEATURE_COLS if col in X.columns]
        if corr_cols:
            corr_source = X[corr_cols].copy()
            corr_source['Risk_Zone_Code'] = y
            corr_source.corr(numeric_only=True).to_csv(corr_path)
    
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
    preds = None
    if os.path.exists(preds_path):
        print("Generating Chart 2: Expert Comparison (Confusion Matrices)")
        preds = pd.read_csv(preds_path)
        model_specs = [
            ('Expert 1: Random Forest', 'y_pred_rf', 'Blues'),
            ('Expert 2: XGBoost', 'y_pred_xgb', 'Oranges'),
            ('Expert 3: LightGBM', 'y_pred_lgbm', 'Greens'),
            ('Expert 4: Neural Network', 'y_pred_nn', 'Purples'),
        ]
        available_specs = [spec for spec in model_specs if spec[1] in preds.columns]

        if available_specs:
            n_models = len(available_specs)
            ncols = 2 if n_models > 2 else n_models
            nrows = int(np.ceil(n_models / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
            axes = np.atleast_1d(axes).flatten()

            for ax, (title, pred_col, cmap) in zip(axes, available_specs):
                cm = confusion_matrix(preds['y_test'], preds[pred_col], labels=[0, 1, 2])
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap=cmap,
                    ax=ax,
                    cbar=False,
                    xticklabels=['Low', 'Medium', 'High'],
                    yticklabels=['Low', 'Medium', 'High']
                )
                ax.set_title(title)
                ax.set_xlabel('Predicted Risk Zone')
                ax.set_ylabel('True Risk Zone')

            for ax in axes[len(available_specs):]:
                ax.axis('off')

        plt.suptitle("Risk Prediction Model Comparison")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(output_dir, '8_2_model_comparison_cm.png'))
        plt.close()

    # 3. Side-by-side model metric comparison
    metrics_df = None
    required_metric_cols = {
        'Accuracy',
        'Macro_Precision',
        'Macro_Recall',
        'Macro_F1',
        'Weighted_F1',
    }
    if os.path.exists(scores_path):
        metrics_df = pd.read_csv(scores_path)
    if (
        metrics_df is None
        or metrics_df.empty
        or not required_metric_cols.issubset(metrics_df.columns)
    ) and preds is not None:
        metrics_df = build_metrics_table(preds)
        if not metrics_df.empty:
            metrics_df.to_csv(scores_path, index=False)

    if metrics_df is not None and not metrics_df.empty:
        print("Generating Chart 3: Model Metric Comparison")
        metric_labels = {
            'Accuracy': 'Accuracy',
            'Macro_Precision': 'Macro Precision',
            'Macro_Recall': 'Macro Recall',
            'Macro_F1': 'Macro F1',
            'Weighted_F1': 'Weighted F1',
        }
        metric_cols = [col for col in metric_labels if col in metrics_df.columns]
        plot_df = metrics_df[['Model'] + metric_cols].melt(
            id_vars='Model',
            value_vars=metric_cols,
            var_name='Metric',
            value_name='Score',
        )
        plot_df['Metric'] = plot_df['Metric'].map(metric_labels)
        model_palette = {
            'Random Forest': '#4c78a8',
            'XGBoost': '#f58518',
            'LightGBM': '#54a24b',
            'Neural Network': '#9c6ade',
        }

        plt.figure(figsize=(14, 7))
        ax = sns.barplot(
            data=plot_df,
            x='Metric',
            y='Score',
            hue='Model',
            palette=model_palette,
            hue_order=[model for model in model_palette if model in plot_df['Model'].unique()],
        )
        ax.set_ylim(0, 1)
        ax.set_title('Model Scorecard Comparison')
        ax.set_xlabel('')
        ax.set_ylabel('Score')
        ax.legend(title='Model', loc='lower right')
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=2, fontsize=8, rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '8_4_model_metrics_comparison.png'))
        plt.close()

    # 4. Correlation matrix over interpretable numeric / engineered features
    if os.path.exists(corr_path):
        print("Generating Chart 4: Feature Correlation Matrix")
        corr = pd.read_csv(corr_path, index_col=0)
        display_labels = [mark_engineered_label(col) for col in corr.columns]
        plt.figure(figsize=(14, 11))
        sns.heatmap(
            corr,
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            xticklabels=display_labels,
            yticklabels=display_labels,
            cbar_kws={'shrink': 0.8}
        )
        plt.title('Feature Correlation Matrix\n(* = engineered, computed before supervised training)')
        formula_note = build_formula_note_from_labels(corr.columns.tolist())
        if formula_note:
            plt.figtext(0.01, 0.01, formula_note, ha='left', va='bottom', fontsize=8)
        plt.tight_layout(rect=[0, 0.10, 1, 0.96])
        plt.savefig(os.path.join(output_dir, '8_3_correlation_heatmap.png'))
        plt.close()

    # 5. Multi-model feature importance comparison
    if os.path.exists(fi_all_models_path):
        print("Generating Chart 5: Multi-Model Feature Importance Comparison")
        fi_all = pd.read_csv(fi_all_models_path)
        top_by_model = fi_all.groupby('Model', group_keys=False).head(8)

        models = top_by_model['Model'].unique().tolist()
        fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 6), squeeze=False)
        axes = axes.flatten()

        palettes = {
            'Random Forest': 'Blues_r',
            'XGBoost': 'Oranges_r',
            'LightGBM': 'Greens_r',
        }

        for ax, model_name in zip(axes, models):
            model_df = top_by_model[top_by_model['Model'] == model_name].sort_values('Importance', ascending=True)
            model_df = model_df.copy()
            model_df['Label'] = model_df.apply(format_feature_label, axis=1)
            sns.barplot(
                data=model_df,
                x='Importance',
                y='Label',
                ax=ax,
                palette=palettes.get(model_name, 'viridis'),
                hue='Label',
                legend=False
            )
            ax.set_title(f"{model_name}\n(* = engineered)")
            ax.set_xlabel('Importance')
            ax.set_ylabel('')

        formula_note = build_formula_note(top_by_model)
        plt.suptitle("Top Features By Model Family")
        if formula_note:
            plt.figtext(0.01, 0.01, formula_note, ha='left', va='bottom', fontsize=8)
        plt.tight_layout(rect=[0, 0.18, 1, 0.95])
        plt.savefig(os.path.join(output_dir, '8_3_feature_importance.png'))
        plt.close()
        
    print("All visualizations successfully generated in output/ folder!")

if __name__ == "__main__":
    run()
