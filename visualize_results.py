import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

# Load data
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
cap_value = y.quantile(0.95)
y = y.clip(upper=cap_value)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load best model
model = joblib.load('best_model.pkl')
y_pred = model.predict(X_test)

# 1. Distribution of Resolution Times
print("\n1. Creating resolution time distribution plot...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['Resolution_Days'], bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Resolution Time (days)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Bug Resolution Times')
axes[0].axvline(df['Resolution_Days'].mean(), color='red', linestyle='--', label=f'Mean: {df["Resolution_Days"].mean():.1f}')
axes[0].axvline(df['Resolution_Days'].median(), color='green', linestyle='--', label=f'Median: {df["Resolution_Days"].median():.1f}')
axes[0].legend()

axes[1].boxplot([df[df['Priority'] == p]['Resolution_Days'].dropna() for p in ['Low', 'Normal', 'High', 'Urgent']])
axes[1].set_xticklabels(['Low', 'Normal', 'High', 'Urgent'])
axes[1].set_ylabel('Resolution Time (days)')
axes[1].set_title('Resolution Time by Priority')

plt.tight_layout()
plt.savefig('1_resolution_distribution.png', dpi=300, bbox_inches='tight')
print("   âœ“ Saved: 1_resolution_distribution.png")

# 2. Actual vs Predicted
print("\n2. Creating actual vs predicted plot...")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, s=20)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Resolution Time (days)')
plt.ylabel('Predicted Resolution Time (days)')
plt.title('Actual vs Predicted Resolution Time')
plt.tight_layout()
plt.savefig('2_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
print("   âœ“ Saved: 2_actual_vs_predicted.png")

# 3. Residuals
print("\n3. Creating residuals plot...")
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5, s=20)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Resolution Time (days)')
plt.ylabel('Residuals (days)')
plt.title('Residual Plot')
plt.tight_layout()
plt.savefig('3_residuals.png', dpi=300, bbox_inches='tight')
print("   âœ“ Saved: 3_residuals.png")

# 4. Feature Importance
if hasattr(model, 'feature_importances_'):
    print("\n4. Creating feature importance plot...")
    importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance for Resolution Time Prediction')
    plt.tight_layout()
    plt.savefig('4_feature_importance.png', dpi=300, bbox_inches='tight')
    print("   âœ“ Saved: 4_feature_importance.png")

# 5. Model Comparison
print("\n5. Creating model comparison plot...")
results_df = pd.read_csv('model_results.csv')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].barh(results_df['Model'], results_df['Test_MAE'])
axes[0].set_xlabel('Mean Absolute Error (days)')
axes[0].set_title('Model Comparison - Test MAE')
axes[0].invert_yaxis()

axes[1].barh(results_df['Model'], results_df['Test_R2'])
axes[1].set_xlabel('RÂ² Score')
axes[1].set_title('Model Comparison - Test RÂ²')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('5_model_comparison.png', dpi=300, bbox_inches='tight')
print("   âœ“ Saved: 5_model_comparison.png")

# 6. Temporal Trends
print("\n6. Creating temporal trends plot...")
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

yearly = df.groupby('Created_Year')['Resolution_Days'].agg(['mean', 'median', 'count'])
axes[0].plot(yearly.index, yearly['mean'], marker='o', label='Mean', linewidth=2)
axes[0].plot(yearly.index, yearly['median'], marker='s', label='Median', linewidth=2)
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Resolution Time (days)')
axes[0].set_title('Average Resolution Time by Year')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].bar(yearly.index, yearly['count'], alpha=0.7)
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Number of Bugs')
axes[1].set_title('Bug Count by Year')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('6_temporal_trends.png', dpi=300, bbox_inches='tight')
print("   âœ“ Saved: 6_temporal_trends.png")

# 7. Category Analysis
print("\n7. Creating category analysis plot...")
categories = ['is_security', 'is_performance', 'is_documentation', 'is_test',
              'is_repair', 'is_compaction', 'is_streaming', 'is_config']

category_stats = []
for cat in categories:
    with_cat = df[df[cat] == 1]['Resolution_Days'].mean()
    without_cat = df[df[cat] == 0]['Resolution_Days'].mean()
    count = df[cat].sum()
    category_stats.append({
        'Category': cat.replace('is_', '').title(),
        'With': with_cat,
        'Without': without_cat,
        'Count': count
    })

cat_df = pd.DataFrame(category_stats).sort_values('With', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

x = np.arange(len(cat_df))
width = 0.35

axes[0].barh(x - width/2, cat_df['With'], width, label='With Category', alpha=0.8)
axes[0].barh(x + width/2, cat_df['Without'], width, label='Without Category', alpha=0.8)
axes[0].set_yticks(x)
axes[0].set_yticklabels(cat_df['Category'])
axes[0].set_xlabel('Average Resolution Time (days)')
axes[0].set_title('Resolution Time by Bug Category')
axes[0].legend()
axes[0].invert_yaxis()

axes[1].barh(cat_df['Category'], cat_df['Count'], alpha=0.7)
axes[1].set_xlabel('Number of Bugs')
axes[1].set_title('Bug Count by Category')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('7_category_analysis.png', dpi=300, bbox_inches='tight')
print("   âœ“ Saved: 7_category_analysis.png")

print("\n" + "="*60)
print("âœ“ All visualizations created successfully!")
print("="*60)
print("\nGenerated files:")
print("  1. 1_resolution_distribution.png")
print("  2. 2_actual_vs_predicted.png")
print("  3. 3_residuals.png")
print("  4. 4_feature_importance.png")
print("  5. 5_model_comparison.png")
print("  6. 6_temporal_trends.png")
print("  7. 7_category_analysis.png")

