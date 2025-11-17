import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*60)
print("ENSEMBLE VISUALIZATION")
print("="*60)

# Load results
ensemble_results = pd.read_csv('ensemble_results.csv')
weights = pd.read_csv('ensemble_weights.csv')

# 1. Model Comparison Bar Chart
print("\n1. Creating model comparison chart...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# MAE Comparison
axes[0].barh(ensemble_results['Approach'], ensemble_results['Test_MAE'], color=['#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
axes[0].set_xlabel('Mean Absolute Error (days)')
axes[0].set_title('Test MAE Comparison')
axes[0].invert_yaxis()

# RMSE Comparison
axes[1].barh(ensemble_results['Approach'], ensemble_results['Test_RMSE'], color=['#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
axes[1].set_xlabel('Root Mean Squared Error (days)')
axes[1].set_title('Test RMSE Comparison')
axes[1].invert_yaxis()

# R² Comparison
axes[2].barh(ensemble_results['Approach'], ensemble_results['Test_R2'], color=['#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
axes[2].set_xlabel('R² Score')
axes[2].set_title('Test R² Comparison')
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig('ensemble_1_comparison.png', dpi=300, bbox_inches='tight')
print("   [OK] Saved: ensemble_1_comparison.png")

# 2. Ensemble Weights Visualization
print("\n2. Creating ensemble weights chart...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Weights pie chart
colors = ['#ff7f0e', '#2ca02c', '#1f77b4']
axes[0].pie(weights['Weight'], labels=weights['Model'], autopct='%1.1f%%', colors=colors, startangle=90)
axes[0].set_title('Weighted Ensemble - Model Contributions')

# Individual MAE comparison
axes[1].bar(weights['Model'], weights['Individual_MAE'], color=colors, alpha=0.7)
axes[1].set_ylabel('Mean Absolute Error (days)')
axes[1].set_title('Individual Model Performance')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('ensemble_2_weights.png', dpi=300, bbox_inches='tight')
print("   [OK] Saved: ensemble_2_weights.png")

# 3. Performance Improvement Chart
print("\n3. Creating improvement chart...")
plt.figure(figsize=(10, 6))

# Calculate improvement percentages
baseline_mae = ensemble_results.iloc[0]['Test_MAE']
improvements = []
for _, row in ensemble_results.iterrows():
    improvement = ((baseline_mae - row['Test_MAE']) / baseline_mae) * 100
    improvements.append(improvement)

ensemble_results['Improvement_%'] = improvements

colors_imp = ['gray' if x <= 0 else 'green' for x in improvements]
plt.barh(ensemble_results['Approach'], ensemble_results['Improvement_%'], color=colors_imp, alpha=0.7)
plt.xlabel('Improvement over Baseline (%)')
plt.title('Performance Improvement Comparison')
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig('ensemble_3_improvement.png', dpi=300, bbox_inches='tight')
print("   [OK] Saved: ensemble_3_improvement.png")

# 4. Ensemble Architecture Diagram (Text-based visualization)
print("\n4. Creating ensemble architecture diagram...")
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# Title
ax.text(0.5, 0.95, 'Stacking Ensemble Architecture', 
        ha='center', va='top', fontsize=16, fontweight='bold')

# Level 0 - Base Models
base_y = 0.7
ax.text(0.5, base_y + 0.1, 'Level 0: Base Models', 
        ha='center', va='center', fontsize=12, fontweight='bold')

base_models = ['Random Forest', 'Gradient Boosting', 'Ridge Regression']
base_x = [0.2, 0.5, 0.8]
for x, model in zip(base_x, base_models):
    # Model box
    rect = plt.Rectangle((x-0.08, base_y-0.05), 0.16, 0.08, 
                         fill=True, facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(x, base_y, model, ha='center', va='center', fontsize=10)
    
    # Arrow down
    ax.arrow(x, base_y-0.05, 0, -0.15, head_width=0.02, head_length=0.02, 
            fc='black', ec='black')

# Predictions layer
pred_y = 0.45
ax.text(0.5, pred_y + 0.1, 'Predictions from Base Models', 
        ha='center', va='center', fontsize=10, style='italic')

for x, model in zip(base_x, base_models):
    ax.text(x, pred_y, 'Pred', ha='center', va='center', 
           bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black'))
    
    # Arrow to meta-model
    ax.arrow(x, pred_y-0.03, (0.5-x)*0.8, -0.1, head_width=0.02, head_length=0.02,
            fc='gray', ec='gray', alpha=0.5)

# Level 1 - Meta Model
meta_y = 0.2
ax.text(0.5, meta_y + 0.1, 'Level 1: Meta-Model', 
        ha='center', va='center', fontsize=12, fontweight='bold')

rect = plt.Rectangle((0.4, meta_y-0.05), 0.2, 0.08, 
                     fill=True, facecolor='lightgreen', edgecolor='black', linewidth=2)
ax.add_patch(rect)
ax.text(0.5, meta_y, 'Linear Regression', ha='center', va='center', fontsize=10)

# Arrow to final prediction
ax.arrow(0.5, meta_y-0.05, 0, -0.08, head_width=0.03, head_length=0.02,
        fc='black', ec='black', linewidth=2)

# Final prediction
final_y = 0.05
rect = plt.Rectangle((0.35, final_y-0.03), 0.3, 0.06, 
                     fill=True, facecolor='gold', edgecolor='black', linewidth=2)
ax.add_patch(rect)
ax.text(0.5, final_y, 'Final Prediction', ha='center', va='center', 
       fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('ensemble_4_architecture.png', dpi=300, bbox_inches='tight')
print("   [OK] Saved: ensemble_4_architecture.png")

# 5. Summary Statistics Table
print("\n5. Creating summary table...")
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

# Prepare table data
table_data = []
table_data.append(['Approach', 'Test MAE', 'Test RMSE', 'Test R²', 'Improvement'])
for _, row in ensemble_results.iterrows():
    table_data.append([
        row['Approach'],
        f"{row['Test_MAE']:.2f} days",
        f"{row['Test_RMSE']:.2f} days",
        f"{row['Test_R2']:.3f}",
        f"{row['Improvement_%']:.2f}%"
    ])

table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                colWidths=[0.35, 0.15, 0.15, 0.15, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header row
for i in range(5):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Highlight best row
table[(1, 0)].set_facecolor('#E8F5E9')
table[(1, 1)].set_facecolor('#E8F5E9')
table[(1, 2)].set_facecolor('#E8F5E9')
table[(1, 3)].set_facecolor('#E8F5E9')
table[(1, 4)].set_facecolor('#E8F5E9')

ax.set_title('Ensemble Methods - Performance Summary', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('ensemble_5_summary_table.png', dpi=300, bbox_inches='tight')
print("   [OK] Saved: ensemble_5_summary_table.png")

print("\n" + "="*60)
print("[OK] All ensemble visualizations created!")
print("="*60)
print("\nGenerated files:")
print("  1. ensemble_1_comparison.png - Model comparison")
print("  2. ensemble_2_weights.png - Ensemble weights")
print("  3. ensemble_3_improvement.png - Performance improvement")
print("  4. ensemble_4_architecture.png - Stacking architecture")
print("  5. ensemble_5_summary_table.png - Results summary")
