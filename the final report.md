<div align="center">

# <span style="color:#1f4e79;">Final Project Report</span>

## <span style="color:#2e6f95;">UK Accident Risk Zone Prediction Using Big Data Analytics and Multi-Model Machine Learning</span>

**Course:** CS588  
**Prepared By:**  
- `[Student 1 Full Name]` - `[A#]`  
- `[Student 2 Full Name]` - `[A#]`  

**Instructor:** `[Instructor Name]`  
**Submission Date:** `[April __, 2026]`

</div>

<div style="padding:10px; background:#f6fbff; border-left:4px solid #1f4e79;">
<strong>Replace the cover-sheet placeholders above before submission.</strong> If you later need a PDF or DOCX, this markdown can be converted outside the repo.
</div>

---

# <span style="color:#1f4e79;">Table of Contents</span>

1. Introduction  
2. Why This Dataset Is Big Data: The 5V Model  
3. Discussion of Methods  
4. Feature Selection, Extraction, and Representation Decisions  
5. Classification, Unsupervised Learning, and the Expert Models  
6. Performance Evaluation Metrics  
7. Results, Data Visualization, and Analysis  
8. Conclusion and Value of the Project  
9. References  
10. Appendix A: Figure Placement Guide  
11. Appendix B: Source Code File Guide  

---

# <span style="color:#1f4e79;">1. Introduction</span>

The goal of this project was to build an end-to-end big data analytics pipeline that uses UK road accident data to predict accident <strong>risk zones</strong> rather than directly predicting police-recorded accident severity. The team hypothesis was:

> Road network characteristics, environmental conditions, and temporal context contain enough signal to classify whether an accident belongs to a low-risk, medium-risk, or high-risk accident zone.

The dataset came from UK STATS19-style road accident records and contained <strong>1,780,653 rows</strong> and <strong>32 columns</strong> in the main accidents table before downstream filtering. Instead of using raw accident severity as the final label, the pipeline first created spatial clusters from latitude and longitude, measured accident density inside those clusters, and converted that density into a new target called <code>Risk_Zone</code> with three classes:

- `Low Risk`
- `Medium Risk`
- `High Risk`

This design let the project study broader environmental risk patterns rather than only the severity of one individual crash. After preprocessing, feature engineering, duplicate pruning, and balancing, the supervised learning stage trained multiple models and compared their predictive performance.

The final submitted modeling setup used:

- a balanced dataset of `120,000` rows,
- an `80/20` stratified holdout split,
- leakage-aware target encoding for high-cardinality road identifiers,
- three primary tree-based "expert" models,
- and one neural-network benchmark after removing a slower stacking ensemble.

---

# <span style="color:#1f4e79;">2. Why This Dataset Is Big Data: The 5V Model</span>

## <span style="color:#2e6f95;">2.1 Volume</span>

This project clearly satisfies the <strong>Volume</strong> characteristic of big data. The raw accidents table alone contained:

- `1,780,653` accident records
- `32` columns before filtering

In addition, the repository also includes related casualties and vehicles files, which shows that the broader domain is large and multi-table in nature. Even after the project narrowed the scope to the accidents file for this implementation, the dataset was still large enough to require careful filtering, column selection, efficient clustering, and memory-aware processing.

## <span style="color:#2e6f95;">2.2 Velocity</span>

Although this project used historical batch data rather than a live streaming feed, road accident reporting is a continuously produced real-world public data stream. New accident records are generated over time by police reporting systems. Therefore, the project addressed <strong>Velocity</strong> in a practical batch-analytics sense: the pipeline was designed as a reusable sequence of scripts that can be rerun on newly updated files.

## <span style="color:#2e6f95;">2.3 Variety</span>

The dataset has strong <strong>Variety</strong> because it combines many different types of variables:

- spatial data: `Latitude`, `Longitude`
- temporal data: `Time`, `Day_of_Week`
- road network data: `Speed_limit`, `Road_Type`, road classes and road numbers
- environmental and scene-condition data: weather, light, road surface, special conditions, carriageway hazards
- traffic and context data: urban/rural area, pedestrian crossing controls
- mixed data types: continuous, ordinal, coded categorical, and engineered interaction features

This variety is one reason the project benefited from a combination of unsupervised learning, feature engineering, encoding, and multiple supervised model families.

## <span style="color:#2e6f95;">2.4 Veracity</span>

The dataset also demonstrates the <strong>Veracity</strong> challenge. The raw data included:

- coded sentinel values such as `-1`
- missing latitude and longitude values
- missing time values
- coded categorical fields that required interpretation
- inconsistent or invalid values such as out-of-range speed limits

Because of this, a major part of the project was devoted to safe cleansing rather than simply throwing the raw CSV into a model.

## <span style="color:#2e6f95;">2.5 Value</span>

The <strong>Value</strong> of the project is that it turns raw road accident records into decision-support insight. Possible use cases include:

- identifying high-risk traffic environments,
- understanding which road and contextual factors are most associated with risky zones,
- supporting future resource allocation or safety planning,
- and demonstrating how a full big-data pipeline can move from raw data to deployable inference.

---

# <span style="color:#1f4e79;">3. Discussion of Methods</span>

This project was organized as a stage-based pipeline. Each stage had a specific purpose and a specific design decision behind it.

## <span style="color:#2e6f95;">3.1 Step 3: Data Acquisition and Filtering</span>

The script `3_data_acquisition_filtering.py` handled the first pass over the raw accidents CSV.

Key decisions:

- The full accidents dataset was loaded from `Accidents0515.csv`.
- Rows missing `Accident_Severity` were dropped as a source-quality check.
- Columns with more than `40%` missing values were eligible for removal.
- However, downstream pipeline-required columns were protected from being silently dropped, even if they exceeded the missing-data threshold.

Why this mattered:

- It reduced memory pressure early.
- It preserved the columns needed later for clustering, feature engineering, and model training.
- It kept `Accident_Severity` for audit/reference, even though it was no longer the final target.

## <span style="color:#2e6f95;">3.2 Step 4: Data Extraction</span>

The script `4_data_extraction.py` selected a focused schema from the larger filtered dataset.

Key decisions:

- Keep only the columns needed for the current risk-zone workflow.
- Preserve geography (`Latitude`, `Longitude`) for spatial clustering in Step 6.
- Preserve `Time` and `Day_of_Week` for temporal feature engineering.
- Retain road identifiers and scene-condition fields.
- Intentionally omit fields that were not part of the active feature plan.

This step functioned like a schema handoff from raw source data into the modeling pipeline.

## <span style="color:#2e6f95;">3.3 Step 5: Data Validation and Cleansing</span>

The script `5_data_validation_cleansing.py` contained several important corrections that improved data quality and reduced avoidable noise.

Major cleansing decisions:

- Replace `-1` only in known UK DfT coded columns where `-1` means unknown or missing.
- Do <strong>not</strong> replace `-1` globally, because doing so could corrupt valid geographic values such as longitude.
- Preserve nulls in `Latitude`, `Longitude`, and `Time` instead of forcing median or mode imputation.
- Impute missing coded variables with mode where appropriate.
- Validate `Speed_limit` values and replace invalid values using the modal valid speed rather than forcing them all to `70`.

Why these choices were important:

- Median-imputing coordinates would create fake accident locations.
- Global `-1` replacement could destroy legitimate western longitudes.
- Preserving real missingness made downstream handling safer and more honest.

## <span style="color:#2e6f95;">3.4 Step 6: Risk-Zone Creation and Aggregation</span>

The script `6_data_aggregation_representation.py` is where the final modeling target was created.

### <span style="color:#3f7f4c;">Unsupervised target construction</span>

The project used `MiniBatchKMeans` with `250` clusters on latitude and longitude.

Reason:

- The raw dataset was too large for slower spatial clustering methods in this project context.
- MiniBatch KMeans scales better to large data while still creating meaningful geographic partitions.

Then the number of accidents in each cluster was counted, and cluster density was converted into a `Risk_Zone` label:

- `Low` if cluster count <= 35th percentile
- `Medium` if cluster count is between the 45th and 75th percentiles
- `High` if cluster count >= 85th percentile
- the middle "gap zones" were discarded to make class boundaries cleaner

This decision is central to the entire project. The final target is therefore <strong>engineered</strong>, not directly copied from the police severity label.

### <span style="color:#3f7f4c;">Dropping direct geography before supervised learning</span>

After `Risk_Zone` was created:

- `Latitude`
- `Longitude`
- `Cluster`
- `Accident_Severity`
- `Time`
- and some count/context fields

were dropped before final supervised model training.

Reason:

- The goal was for the classifier to learn environmental and road-context patterns, not trivially memorize raw coordinates.

### <span style="color:#3f7f4c;">Duplicate pruning</span>

One of the final improvements was to remove exact duplicate `feature + target` rows before balancing.

This mattered because once geography was dropped, many accidents became identical from the model's perspective. Keeping many repeated copies would make the holdout split easier than it should be.

Observed result:

- `310,481` exact duplicate supervised rows were removed before balancing.

### <span style="color:#3f7f4c;">Balancing</span>

The final supervised dataset was balanced to:

- `40,000` `Low Risk`
- `40,000` `Medium Risk`
- `40,000` `High Risk`

for a final training table of:

- `120,000` rows
- `25` raw/engineered predictor columns before target encoding expansion

Balancing ensured that accuracy and macro metrics would not be dominated by the largest class.

---

# <span style="color:#1f4e79;">4. Feature Selection, Extraction, and Representation Decisions</span>

## <span style="color:#2e6f95;">4.1 Selected Base Features</span>

The final selected base features came from four main categories:

- road structure and hierarchy
- traffic environment and area type
- environmental conditions
- temporal context

Examples include:

- `Speed_limit`
- `Road_Type`
- `1st_Road_Class`
- `1st_Road_Number`
- `2nd_Road_Class`
- `2nd_Road_Number`
- `Junction_Detail`
- `Light_Conditions`
- `Weather_Conditions`
- `Road_Surface_Conditions`
- `Day_of_Week`
- `Urban_or_Rural_Area`

## <span style="color:#2e6f95;">4.2 Engineered Features</span>

The project added several engineered features to improve representational power:

| Engineered Feature | Formula / Construction | Why It Was Added |
|---|---|---|
| `Hour` | Extract hour from `Time` | Convert raw time string into a usable numeric signal |
| `IsNight` | `1 if Hour < 6 or Hour >= 20 else 0` | Capture day/night traffic differences |
| `IsRushHour` | `1 if Hour in {7,8,9,17,18,19} else 0` | Represent commuter traffic periods |
| `Hour_Sin` | `sin(2*pi*Hour/24)` | Preserve circular time structure |
| `Hour_Cos` | `cos(2*pi*Hour/24)` | Pair with sine for clock-face encoding |
| `DayNight_Context` | `str(Day_of_Week) + "_" + str(IsNight)` | Capture interactions such as weekend nights |
| `Urban_Speed_Net` | `Urban_or_Rural_Area * Speed_limit` | Combine speed and area context |
| `Junction_Complexity` | `Junction_Detail + Junction_Control` | Approximate interaction complexity around junctions |

These engineered features were later marked in the feature-importance outputs so the report could distinguish them from raw source variables.

## <span style="color:#2e6f95;">4.3 Feature Representation and Encoding</span>

Two kinds of encoding were used:

### Label encoding

Low-cardinality string/categorical columns were label-encoded in Step 6 to make them compatible with the ML libraries.

### Leakage-aware target encoding

High-cardinality columns such as:

- `1st_Road_Number`
- `2nd_Road_Number`
- `DayNight_Context`

were target-encoded in Step 7 using out-of-fold encoding. This was a strong design decision because it:

- reduced the dimensional burden of very high-cardinality identifiers,
- preserved predictive signal,
- and avoided direct target leakage from using the whole training fold at once.

## <span style="color:#2e6f95;">4.4 Dimensionality Reduction</span>

Classical dimensionality reduction such as PCA was <strong>not</strong> used.

Reason:

- the final feature set was already compact enough,
- interpretability mattered more than projection into latent components,
- and tree-based models handle mixed structured features well without PCA.

In this project, feature extraction and representation learning were more valuable than formal dimensionality reduction.

---

# <span style="color:#1f4e79;">5. Classification, Unsupervised Learning, and the Expert Models</span>

## <span style="color:#2e6f95;">5.1 Unsupervised Learning</span>

Unsupervised learning occurred first through spatial clustering:

- `MiniBatchKMeans`
- `250` geographic clusters
- cluster density converted into three risk classes

This step transformed raw accident locations into a machine-learning target.

## <span style="color:#2e6f95;">5.2 The Three Primary Expert Models</span>

The three main supervised "experts" in the final report were:

1. **Random Forest**  
   A bagging-based tree ensemble that is robust, easy to interpret with feature importance, and a strong baseline for tabular data.

2. **XGBoost**  
   A boosted-tree model that usually performs very well on structured classification tasks and ultimately gave the best holdout result in this project.

3. **LightGBM**  
   Another gradient boosting system optimized for efficiency and strong performance on large tabular datasets.

## <span style="color:#2e6f95;">5.3 Additional Neural-Network Benchmark</span>

A neural network benchmark was also added:

- `MLPClassifier`
- hidden layers `(128, 64)`
- `ReLU` activation
- `Adam` optimizer
- early stopping enabled

Why it was included:

- The previous stacking ensemble was removed because it was slower and did not produce enough extra value.
- The neural network provided a useful non-tree comparison.

## <span style="color:#2e6f95;">5.4 Final Modeling Decision</span>

The final training flow was:

1. stratified `80/20` holdout split  
2. overlap check on train/test feature rows  
3. out-of-fold target encoding on selected high-cardinality features  
4. train the four supervised models  
5. compare them using accuracy, precision, recall, and F1  
6. select the best-performing model and save a deployable inference bundle

Observed overlap after duplicate pruning:

- `208` exact feature rows out of `24,000` test rows still overlapped between train and test

This was presented honestly as a limitation rather than hidden.

---

# <span style="color:#1f4e79;">6. Performance Evaluation Metrics</span>

Because this was a multiclass classification project, the main evaluation metrics were:

- `Accuracy`
- `Macro Precision`
- `Macro Recall`
- `Macro F1`
- `Weighted F1`

These metrics are appropriate because:

- accuracy shows overall correctness,
- macro metrics treat each class equally,
- weighted F1 accounts for class contribution while still reflecting classification quality.

`RMSE` and `SNR` were not used because this was not a regression or signal-processing problem.

## <span style="color:#2e6f95;">Final Holdout Results</span>

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.7761 | 0.7758 | 0.7761 | 0.7756 | 0.7756 |
| LightGBM | 0.7740 | 0.7738 | 0.7740 | 0.7736 | 0.7736 |
| Random Forest | 0.7675 | 0.7677 | 0.7675 | 0.7671 | 0.7671 |
| Neural Network | 0.7480 | 0.7482 | 0.7480 | 0.7479 | 0.7479 |

Main observation:

- `XGBoost` was the best final model on the standard holdout split.
- `LightGBM` was very close behind.
- The neural network was competitive but did not surpass the tree-based methods.

---

# <span style="color:#1f4e79;">7. Results, Data Visualization, and Analysis</span>

## <span style="color:#2e6f95;">7.1 Key Results</span>

The final results support the project hypothesis. Even after geography was removed from the supervised feature set, the remaining road, environment, and temporal variables still carried enough signal to classify accidents into learned risk-zone categories with meaningful accuracy.

The most important findings were:

- risk zones can be learned from non-coordinate contextual features,
- boosted-tree methods performed best,
- medium-risk zones were consistently harder to classify than low- or high-risk zones,
- engineered context features such as `Urban_Speed_Net` contributed strong value,
- and high-cardinality road identifiers retained strong signal after leakage-aware target encoding.

## <span style="color:#2e6f95;">7.2 Correlation Matrix Analysis</span>

The correlation heatmap revealed several expected relationships:

- `Speed_limit` and `Urban_Speed_Net` had a very strong positive correlation (`0.9377`) because the latter is derived from the former.
- `Urban_or_Rural_Area` and `Urban_Speed_Net` were also strongly correlated (`0.8733`).
- `Hour` and `Hour_Sin` showed strong circular-encoding structure (`-0.8170`).
- `IsNight` and `Hour_Cos` were strongly related (`0.8149`), which makes sense because both capture night-time position on the daily cycle.
- `Urban_or_Rural_Area` and `Risk_Zone_Code` had a moderately strong negative relationship (`-0.5101`), suggesting that urban versus rural context is an important differentiator of learned risk zones.

These relationships were meaningful rather than accidental because many of them came directly from deliberate feature engineering.

## <span style="color:#2e6f95;">7.3 Confusion Matrix Analysis</span>

The confusion matrices showed a consistent pattern across all four supervised models:

- `Low Risk` and `High Risk` were easier to separate than `Medium Risk`.
- `Medium Risk` had the lowest precision and recall.
- This is reasonable because the medium class sits between the two extremes and is naturally the most ambiguous zone.

This pattern increases confidence that the models are learning a sensible class structure instead of producing random behavior.

## <span style="color:#2e6f95;">7.4 Feature Importance Analysis</span>

The combined feature-importance figure showed that the following features were repeatedly influential:

- target-encoded versions of `1st_Road_Number`
- target-encoded versions of `2nd_Road_Number`
- `Urban_or_Rural_Area`
- `Urban_Speed_Net`
- `1st_Road_Class`

Interpretation:

- Road identity and road hierarchy matter strongly.
- Area context matters strongly.
- The interaction between area type and speed limit captures important risk structure.

Model-specific observations:

- Random Forest and LightGBM placed especially strong weight on target-encoded road-number features.
- XGBoost gave the largest single-model emphasis to `Urban_or_Rural_Area`, followed by `Urban_Speed_Net`.

## <span style="color:#2e6f95;">7.5 Data Visualization Summary</span>

The most useful visuals for the main body of the report are:

- spatial risk cluster map
- model score comparison plot
- confusion matrix comparison
- combined feature-importance plot
- correlation heatmap

If page space becomes tight, the feature-importance chart and correlation heatmap can be moved to the appendix while keeping the score comparison and confusion matrix in the main results section.

---

# <span style="color:#1f4e79;">8. Conclusion and Value of the Project</span>

## <span style="color:#2e6f95;">8.1 Conclusion of the Hypothesis</span>

The project hypothesis was supported.

It is possible to use road-network, environmental, and temporal features to predict whether an accident belongs to a low-risk, medium-risk, or high-risk spatial zone. The best final model, XGBoost, achieved approximately `77.61%` accuracy with similarly strong macro F1 on the balanced holdout set.

The project also showed that:

- feature engineering materially improved model quality,
- high-cardinality road features can be handled effectively with leakage-aware target encoding,
- and tree-based ensemble methods are particularly strong for this kind of structured risk classification task.

## <span style="color:#2e6f95;">8.2 Practical Value</span>

The value of this project is not only the final accuracy number. It also demonstrates:

- a complete big-data workflow,
- conversion of raw geographic records into usable risk categories,
- reusable inference tooling (`10_inference.py`, `predict_server.py`, and the web UI),
- and interpretable visual outputs that explain why the models behave as they do.

## <span style="color:#2e6f95;">8.3 Human Contribution vs. AI Contribution</span>

This project was a collaboration between human team members and AI-assisted development, but the <strong>human contribution remained essential</strong>.

### Human team contribution

- defining the problem worth solving,
- deciding to pivot from raw severity prediction to risk-zone prediction,
- judging which features and results added value to the hypothesis,
- identifying red flags in cleaning and evaluation,
- deciding when to accept a practical class-project tradeoff versus a stricter experimental redesign,
- and shaping the final story for presentation and reporting

### AI contribution

- accelerating implementation across multiple scripts,
- drafting and refactoring preprocessing and training code,
- adding target encoding, temporal features, visualizations, and inference support,
- helping compare alternative modeling approaches quickly,
- and assisting with documentation and reporting

### Final value statement

The strongest value came from the combination:

- the human team provided domain judgment, prioritization, and research framing,
- while AI accelerated coding, iteration speed, and documentation support.

In other words, AI increased execution speed, but the humans defined what the project meant and which conclusions were trustworthy enough to present.

---

# <span style="color:#1f4e79;">9. References</span>

1. UK Department for Transport road accident data (`Accidents0515.csv`, `Casualties0515.csv`, `Vehicles0515.csv`).  
2. Pedregosa, F. et al. *Scikit-learn: Machine Learning in Python*.  
3. Chen, T. and Guestrin, C. *XGBoost: A Scalable Tree Boosting System*.  
4. Ke, G. et al. *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*.  
5. `category_encoders` Python package for target encoding.  
6. `pandas`, `numpy`, `matplotlib`, and `seaborn` Python libraries used in preprocessing and visualization.  

---

# <span style="color:#1f4e79;">10. Appendix A: Figure Placement Guide</span>

Use the existing PNG files from the `output/` folder in the following locations.

| Figure | File | Best Report Section | What to Say About It |
|---|---|---|---|
| Figure 1 | `output/8_1_spatial_risk_map.png` | End of Methods or start of Results | Show how geographic clustering produced Low/Medium/High risk zones. Focus on the spatial separation of dense vs. sparse accident regions. |
| Figure 2 | `output/8_4_model_metrics_comparison.png` | Main Results section | Use this as the primary model-comparison figure. Highlight that XGBoost slightly outperformed LightGBM and Random Forest across accuracy and F1. |
| Figure 3 | `output/8_2_model_comparison_cm.png` | Results, immediately after Figure 2 | Discuss that the middle class (`Medium Risk`) is the hardest class, while `Low Risk` and `High Risk` have stronger diagonals. |
| Figure 4 | `output/8_3_feature_importance.png` | Analysis section | Focus on repeated top features across models: road-number encodings, `Urban_or_Rural_Area`, `Urban_Speed_Net`, and `1st_Road_Class`. |
| Figure 5 | `output/8_3_correlation_heatmap.png` | Data Visualization section or Appendix if space is tight | Point out the intentionally strong correlations created by engineered features such as `Urban_Speed_Net` and cyclical hour encoding. |

Recommended order in the main report:

1. Figure 2: model score comparison  
2. Figure 3: confusion matrices  
3. Figure 1: spatial risk map  
4. Figure 4: feature importance  
5. Figure 5: correlation heatmap  

If the report must stay near 10 pages, Figures 4 and 5 can be moved to the appendix without hurting the main argument.

---

# <span style="color:#1f4e79;">11. Appendix B: Source Code File Guide</span>

| File | Purpose |
|---|---|
| `3_data_acquisition_filtering.py` | Loads the raw accidents file, applies early filtering, and preserves pipeline-required columns. |
| `4_data_extraction.py` | Selects the subset of columns needed for the current risk-zone workflow. |
| `5_data_validation_cleansing.py` | Performs safe missing-value handling, sentinel replacement, and speed-limit validation. |
| `6_data_aggregation_representation.py` | Builds spatial clusters, derives the `Risk_Zone` target, engineers new features, removes duplicates, balances classes, and saves final training arrays. |
| `7_data_analysis.py` | Splits the data, applies leakage-aware target encoding, trains the supervised models, compares their metrics, and saves the best model bundle. |
| `8_data_visualization.py` | Generates the spatial risk map, confusion matrices, model score plot, correlation heatmap, and feature-importance chart. |
| `risk_zone_model.py` | Shared inference/preprocessing layer used by the CLI and web server. |
| `10_inference.py` | Command-line prediction interface for the trained risk-zone model bundle. |
| `predict_server.py` | Local web/API server exposing `/predict`, `/schema`, and `/healthz`. |
| `index.html` | Browser UI for testing custom prediction scenarios. |
| `run_pipeline.sh` | Convenience script for running the pipeline end to end. |
| `master_script.py` | Pipeline orchestrator / project runner helper. |

## <span style="color:#2e6f95;">Suggested Submission Packaging</span>

For Canvas submission, the final deliverables should be packaged as:

- report
- presentation
- source code

inside a single ZIP named in the required format:

- `Student_firstname_lastname.zip`

If you export this markdown to PDF or DOCX later, rename the final report file to match the same submission naming convention.
