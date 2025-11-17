import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("BUG RESOLUTION TIME PREDICTION - MODEL TRAINING")
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

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0),
    'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
}

# Train and evaluate models
print("\n" + "="*60)
print("MODEL TRAINING & EVALUATION")
print("="*60)

results = []

for name, model in models.items():
    print(f"\n{name}:")
    print("-" * 40)
    
    # Use scaled data for linear models, original for tree-based
    if 'Regression' in name:
        X_train_use = X_train_scaled
        X_test_use = X_test_scaled
    else:
        X_train_use = X_train
        X_test_use = X_test
    
    # Train
    model.fit(X_train_use, y_train)
    
    # Predict
    y_pred_train = model.predict(X_train_use)
    y_pred_test = model.predict(X_test_use)
    
    # Evaluate
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"  Train MAE: {train_mae:.2f} days")
    print(f"  Test MAE:  {test_mae:.2f} days")
    print(f"  Train RMSE: {train_rmse:.2f} days")
    print(f"  Test RMSE:  {test_rmse:.2f} days")
    print(f"  Train RÂ²: {train_r2:.3f}")
    print(f"  Test RÂ²:  {test_r2:.3f}")
    
    results.append({
        'Model': name,
        'Train_MAE': train_mae,
        'Test_MAE': test_mae,
        'Train_RMSE': train_rmse,
        'Test_RMSE': test_rmse,
        'Train_R2': train_r2,
        'Test_R2': test_r2
    })

# Results summary
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Test_MAE')
print("\nModels ranked by Test MAE (lower is better):")
print(results_df.to_string(index=False))

# Save best model
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]

if 'Regression' in best_model_name:
    X_train_use = X_train_scaled
else:
    X_train_use = X_train

best_model.fit(X_train_use, y_train)
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print(f"\nâœ“ Best model ({best_model_name}) saved to 'best_model.pkl'")

# Feature importance (for tree-based models)
if hasattr(best_model, 'feature_importances_'):
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE (Top 10)")
    print("="*60)
    
    importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(importance_df.head(10).to_string(index=False))

# Save results
results_df.to_csv('model_results.csv', index=False)
print("\nâœ“ Results saved to 'model_results.csv'")

print("\n" + "="*60)
print("BASELINE COMPARISON")
print("="*60)

# Baseline: predict mean
baseline_pred = np.full(len(y_test), y_train.mean())
baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))

print(f"\nBaseline (predict mean): {y_train.mean():.2f} days")
print(f"  Baseline MAE: {baseline_mae:.2f} days")
print(f"  Baseline RMSE: {baseline_rmse:.2f} days")

best_test_mae = results_df.iloc[0]['Test_MAE']
improvement = ((baseline_mae - best_test_mae) / baseline_mae) * 100

print(f"\nBest model improvement over baseline: {improvement:.1f}%")

print("\nâœ“ Training complete!")

