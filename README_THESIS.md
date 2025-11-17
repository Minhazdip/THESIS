# Bug Resolution Time Prediction - Thesis Project

## Overview
This project predicts the resolution time of Apache Cassandra bugs using machine learning techniques. The dataset contains 4,612 bug reports from 2020-2024.

## Research Questions
1. **Primary**: Can we accurately predict how long it will take to resolve a bug based on its characteristics?
2. **Secondary**: What factors most significantly influence bug resolution time?
3. **Exploratory**: How do different bug categories and priorities affect resolution time?

## Dataset
- **Source**: Apache Cassandra JIRA bug tracking system
- **Size**: 4,612 bug reports
- **Time Period**: 2020-2024
- **Features**: Summary, Description, Priority, Status, Creation Date, Resolution Date

## Methodology

### 1. Data Preprocessing
- Parse dates and calculate resolution time in days
- Handle missing values
- Remove outliers (cap at 95th percentile)
- Extract text features (length, keyword counts)

### 2. Feature Engineering
**Text Features:**
- Summary length
- Description length
- Keyword count
- Has description flag

**Temporal Features:**
- Year created
- Month created
- Day of week created

**Category Features (Binary):**
- Security issues
- Performance issues
- Documentation issues
- Test-related issues
- Repair issues
- Compaction issues
- Streaming issues
- Configuration issues

**Encoded Features:**
- Priority (Low, Normal, High, Urgent)
- Status

### 3. Models Evaluated

**Single Models:**
1. Linear Regression (baseline)
2. Ridge Regression
3. Lasso Regression
4. Decision Tree
5. Random Forest
6. Gradient Boosting

**Ensemble Models (Multiple Models Combined):**
7. Stacking Ensemble (RF + GB + Ridge → Meta-Model)
8. Voting Ensemble (Simple averaging)
9. Weighted Ensemble (Performance-based weighting)

### 4. Evaluation Metrics
- **MAE (Mean Absolute Error)**: Average prediction error in days
- **RMSE (Root Mean Squared Error)**: Penalizes large errors
- **R² Score**: Proportion of variance explained

## Project Structure
```
├── cassandra_bugs.csv              # Original dataset
├── bug_resolution_analysis.py      # Data preprocessing & EDA
├── train_models.py                 # Model training & evaluation
├── visualize_results.py            # Generate charts
├── predict_new_bug.py              # Prediction tool for new bugs
├── processed_bugs.csv              # Processed dataset (generated)
├── model_results.csv               # Model comparison (generated)
├── best_model.pkl                  # Trained model (generated)
└── *.png                           # Visualization outputs
```

## How to Run

### Step 1: Data Analysis & Preprocessing
```bash
python bug_resolution_analysis.py
```
**Output**: 
- Exploratory analysis results
- `processed_bugs.csv`

### Step 2: Train Models
```bash
python train_models.py
```
**Output**:
- Model comparison results
- `best_model.pkl`
- `model_results.csv`

### Step 3: Generate Visualizations
```bash
python visualize_results.py
```
**Output**: 7 visualization files

### Step 4: Predict New Bugs
```bash
python predict_new_bug.py
```

## Expected Results

### Key Findings
1. **Average Resolution Time**: ~70 days (median: 13 days)
2. **Priority Impact**: 
   - Urgent bugs: ~40 days
   - High priority: ~109 days
   - Normal: ~70 days
   - Low: ~84 days

3. **Category Impact**:
   - Test-related bugs: Most common (1,204 issues)
   - Documentation bugs: Faster resolution
   - Security bugs: Require special attention

### Model Performance (Expected)
- Best model MAE: 20-30 days
- R² Score: 0.3-0.5
- Improvement over baseline: 30-40%

## Thesis Structure Outline

### Chapter 1: Introduction
- Problem statement
- Research objectives
- Significance of the study

### Chapter 2: Literature Review
- Bug tracking systems
- Software defect prediction
- Machine learning in software engineering
- Related work on resolution time prediction

### Chapter 3: Methodology
- Dataset description
- Feature engineering approach
- Model selection rationale
- Evaluation metrics

### Chapter 4: Implementation
- Data preprocessing pipeline
- Feature extraction techniques
- Model training process
- Hyperparameter tuning

### Chapter 5: Results & Analysis
- Exploratory data analysis
- Model comparison
- Feature importance analysis
- Error analysis

### Chapter 6: Discussion
- Interpretation of results
- Practical implications
- Limitations
- Threats to validity

### Chapter 7: Conclusion
- Summary of findings
- Contributions
- Future work

## Key Insights for Thesis

### Interesting Findings
1. **Counterintuitive**: High priority bugs take LONGER than urgent bugs
   - Possible reason: Urgent bugs are simpler/critical path
   - High priority bugs may be more complex

2. **Test Dominance**: 26% of all bugs are test-related
   - Opportunity for automated test improvement

3. **Temporal Trends**: Bug resolution efficiency varies by year
   - Could indicate team size changes or process improvements

4. **Text Length**: Weak correlation with resolution time
   - Suggests description quality > quantity

### Potential Extensions
1. **NLP Enhancement**: Use BERT/transformers for text features
2. **Classification**: Predict resolution time buckets (fast/medium/slow)
3. **Survival Analysis**: Model time-to-resolution as survival problem
4. **Multi-task Learning**: Jointly predict resolution time and priority
5. **Causal Analysis**: Identify causal factors vs correlations

## Requirements
```
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```

Install with:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## Contact & Support
For questions about this thesis project, refer to the code comments or documentation.

## License
Educational use only - Apache Cassandra data is publicly available.
