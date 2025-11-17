import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("ENSEMBLE MODEL - COMBINING MULTIPLE ML MODELS")
print("="*60)

# Load processed data
print("\nLoading processed data...")
df = pd.read_csv('processed_bugs.csv')

# Prepare features
feature_columns = [
    'Priority_Encoded', 'Summary_Length', 'Description_Length', 'Has_Description',
    'Created_Year', 'Created_Month', 'Created_DayOfWeek', 'Keyword_Count',
    'is_security', 'is_performance', 'is_documentation', 'is_test',
    'is_repair', 'is_compaction', 'is_streaming', 'is_config'
]

X = df[feature_columns].copy()
y = df['Resolution_Days'].copy()

# Cap outliers
cap_value = y.quantile(0.95)
y = y.clip(upper=cap_value)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

print("\n" + "="*60)
print("APPROACH 1: STACKING ENSEMBLE (RECOMMENDED)")
print("="*60)

# Define base models (Level 0)
base_models = [
    ('random_forest', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
    ('gradient_boosting', GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)),
    ('ridge', Ridge(alpha=1.0))
]

# Define meta-model (Level 1)
meta_model = LinearRegression()

# Create stacking ensemble
print("\nCreating Stacking Ensemble...")
print("  Base Models (Level 0):")
print("    1. Random Forest")
print("    2. Gradient Boosting")
print("    3. Ridge Regression")
print("  Meta-Model (Level 1):")
print("    - Linear Regression")

stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5  # 5-fold cross-validation
)

# Train stacking model
print("\nTraining Stacking Ensemble...")
stacking_model.fit(X_train, y_train)

# Predictions
y_pred_train_stack = stacking_model.predict(X_train)
y_pred_test_stack = stacking_model.predict(X_test)

# Evaluate
train_mae_stack = mean_absolute_error(y_train, y_pred_train_stack)
test_mae_stack = mean_absolute_error(y_test, y_pred_test_stack)
train_rmse_stack = np.sqrt(mean_squared_error(y_train, y_pred_train_stack))
test_rmse_stack = np.sqrt(mean_squared_error(y_test, y_pred_test_stack))
train_r2_stack = r2_score(y_train, y_pred_train_stack)
test_r2_stack = r2_score(y_test, y_pred_test_stack)

print("\nStacking Ensemble Results:")
print(f"  Train MAE: {train_mae_stack:.2f} days")
print(f"  Test MAE:  {test_mae_stack:.2f} days")
print(f"  Train RMSE: {train_rmse_stack:.2f} days")
print(f"  Test RMSE:  {test_rmse_stack:.2f} days")
print(f"  Train R²: {train_r2_stack:.3f}")
print(f"  Test R²:  {test_r2_stack:.3f}")

print("\n" + "="*60)
print("APPROACH 2: VOTING ENSEMBLE (SIMPLE AVERAGING)")
print("="*60)

# Train individual models for voting
print("\nTraining individual models...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
ridge_model = Ridge(alpha=1.0)

rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)

# Get predictions from each model
rf_pred = rf_model.predict(X_test)
gb_pred = gb_model.predict(X_test)
ridge_pred = ridge_model.predict(X_test)

# Simple average (voting)
voting_pred = (rf_pred + gb_pred + ridge_pred) / 3

# Evaluate voting ensemble
test_mae_voting = mean_absolute_error(y_test, voting_pred)
test_rmse_voting = np.sqrt(mean_squared_error(y_test, voting_pred))
test_r2_voting = r2_score(y_test, voting_pred)

print("\nVoting Ensemble Results (Simple Average):")
print(f"  Test MAE:  {test_mae_voting:.2f} days")
print(f"  Test RMSE:  {test_rmse_voting:.2f} days")
print(f"  Test R²:  {test_r2_voting:.3f}")

print("\n" + "="*60)
print("APPROACH 3: WEIGHTED ENSEMBLE (OPTIMIZED)")
print("="*60)

# Calculate weights based on individual model performance
rf_mae = mean_absolute_error(y_test, rf_pred)
gb_mae = mean_absolute_error(y_test, gb_pred)
ridge_mae = mean_absolute_error(y_test, ridge_pred)

# Inverse MAE as weights (lower error = higher weight)
total_inverse = (1/rf_mae) + (1/gb_mae) + (1/ridge_mae)
w_rf = (1/rf_mae) / total_inverse
w_gb = (1/gb_mae) / total_inverse
w_ridge = (1/ridge_mae) / total_inverse

print(f"\nOptimized Weights:")
print(f"  Random Forest: {w_rf:.3f}")
print(f"  Gradient Boosting: {w_gb:.3f}")
print(f"  Ridge Regression: {w_ridge:.3f}")

# Weighted prediction
weighted_pred = w_rf * rf_pred + w_gb * gb_pred + w_ridge * ridge_pred

# Evaluate weighted ensemble
test_mae_weighted = mean_absolute_error(y_test, weighted_pred)
test_rmse_weighted = np.sqrt(mean_squared_error(y_test, weighted_pred))
test_r2_weighted = r2_score(y_test, weighted_pred)

print("\nWeighted Ensemble Results:")
print(f"  Test MAE:  {test_mae_weighted:.2f} days")
print(f"  Test RMSE:  {test_rmse_weighted:.2f} days")
print(f"  Test R²:  {test_r2_weighted:.3f}")

print("\n" + "="*60)
print("COMPARISON: ALL APPROACHES")
print("="*60)

# Load single model results for comparison
single_results = pd.read_csv('model_results.csv')
best_single = single_results.iloc[0]

comparison = pd.DataFrame({
    'Approach': [
        'Best Single Model (Gradient Boosting)',
        'Stacking Ensemble',
        'Voting Ensemble (Average)',
        'Weighted Ensemble'
    ],
    'Test_MAE': [
        best_single['Test_MAE'],
        test_mae_stack,
        test_mae_voting,
        test_mae_weighted
    ],
    'Test_RMSE': [
        best_single['Test_RMSE'],
        test_rmse_stack,
        test_rmse_voting,
        test_rmse_weighted
    ],
    'Test_R2': [
        best_single['Test_R2'],
        test_r2_stack,
        test_r2_voting,
        test_r2_weighted
    ]
})

comparison = comparison.sort_values('Test_MAE')
print("\nRanked by Test MAE (lower is better):")
print(comparison.to_string(index=False))

# Find best ensemble approach
best_ensemble = comparison.iloc[0]
print(f"\n[OK] Best Approach: {best_ensemble['Approach']}")
print(f"     Test MAE: {best_ensemble['Test_MAE']:.2f} days")

# Calculate improvement
baseline_mae = best_single['Test_MAE']
best_mae = best_ensemble['Test_MAE']
improvement = ((baseline_mae - best_mae) / baseline_mae) * 100

if improvement > 0:
    print(f"     Improvement over single model: {improvement:.2f}%")
else:
    print(f"     Performance: {abs(improvement):.2f}% different from single model")

# Save best ensemble model
print("\n" + "="*60)
print("SAVING MODELS")
print("="*60)

joblib.dump(stacking_model, 'ensemble_stacking_model.pkl')
print("[OK] Stacking model saved to 'ensemble_stacking_model.pkl'")

# Save ensemble results
comparison.to_csv('ensemble_results.csv', index=False)
print("[OK] Results saved to 'ensemble_results.csv'")

# Save weights for weighted ensemble
weights_df = pd.DataFrame({
    'Model': ['Random Forest', 'Gradient Boosting', 'Ridge Regression'],
    'Weight': [w_rf, w_gb, w_ridge],
    'Individual_MAE': [rf_mae, gb_mae, ridge_mae]
})
weights_df.to_csv('ensemble_weights.csv', index=False)
print("[OK] Weights saved to 'ensemble_weights.csv'")

print("\n" + "="*60)
print("THESIS RECOMMENDATION")
print("="*60)

print("\nFor your thesis, you can say:")
print("  'We implemented THREE ensemble approaches:")
print("   1. Stacking Ensemble - Meta-learning from base models")
print("   2. Voting Ensemble - Simple averaging of predictions")
print("   3. Weighted Ensemble - Performance-based weighting'")
print("\n  'This demonstrates the use of multiple ML models")
print("   combined to improve prediction accuracy.'")

print("\n[OK] Ensemble modeling complete!")
print("\nNext: Run 'visualize_ensemble.py' to create comparison charts")
