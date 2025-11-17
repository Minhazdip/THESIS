import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
df = pd.read_csv('cassandra_bugs.csv')

# Parse dates
df['Created'] = pd.to_datetime(df['Created'], format='%d/%b/%y %H:%M', errors='coerce')
df['Resolved'] = pd.to_datetime(df['Resolved'], format='%d/%b/%y %H:%M', errors='coerce')

# Calculate resolution time
df['Resolution_Days'] = (df['Resolved'] - df['Created']).dt.days

# Filter only resolved bugs with valid resolution time
resolved_df = df[df['Resolution_Days'].notna()].copy()
resolved_df = resolved_df[resolved_df['Resolution_Days'] >= 0]  # Remove negative values

print(f"\nTotal resolved bugs: {len(resolved_df)}")
print(f"Resolution time range: {resolved_df['Resolution_Days'].min():.0f} - {resolved_df['Resolution_Days'].max():.0f} days")

# Feature Engineering
print("\n" + "="*60)
print("FEATURE ENGINEERING")
print("="*60)

# 1. Text length features
resolved_df['Summary_Length'] = resolved_df['Summary'].fillna('').str.len()
resolved_df['Description_Length'] = resolved_df['Description'].fillna('').str.len()
resolved_df['Has_Description'] = resolved_df['Description'].notna().astype(int)

# 2. Temporal features
resolved_df['Created_Year'] = resolved_df['Created'].dt.year
resolved_df['Created_Month'] = resolved_df['Created'].dt.month
resolved_df['Created_DayOfWeek'] = resolved_df['Created'].dt.dayofweek

# 3. Category detection
categories = {
    'is_security': ['security', 'cve', 'vulnerability', 'ssl', 'tls', 'encryption'],
    'is_performance': ['performance', 'slow', 'latency', 'optimization', 'memory', 'cpu'],
    'is_documentation': ['doc', 'documentation', 'readme'],
    'is_test': ['test', 'testing', 'junit', 'flaky', 'dtest'],
    'is_repair': ['repair', 'anticompaction'],
    'is_compaction': ['compaction', 'sstable'],
    'is_streaming': ['stream', 'streaming'],
    'is_config': ['config', 'configuration', 'yaml'],
}

for category, keywords in categories.items():
    pattern = '|'.join(keywords)
    resolved_df[category] = (
        resolved_df['Summary'].str.contains(pattern, case=False, na=False) |
        resolved_df['Description'].str.contains(pattern, case=False, na=False)
    ).astype(int)

# 4. Keyword counts
resolved_df['Keyword_Count'] = resolved_df['Summary'].fillna('').str.split().str.len()

# 5. Encode categorical variables
le_priority = LabelEncoder()
le_status = LabelEncoder()

resolved_df['Priority_Encoded'] = le_priority.fit_transform(resolved_df['Priority'])
resolved_df['Status_Encoded'] = le_status.fit_transform(resolved_df['Status'])

print("\nFeatures created:")
print(f"  - Text features: Summary_Length, Description_Length, Has_Description")
print(f"  - Temporal features: Year, Month, DayOfWeek")
print(f"  - Category flags: {len(categories)} categories")
print(f"  - Encoded features: Priority, Status")

# Exploratory Analysis
print("\n" + "="*60)
print("EXPLORATORY ANALYSIS")
print("="*60)

print("\n1. Resolution Time by Priority:")
priority_stats = resolved_df.groupby('Priority')['Resolution_Days'].agg(['mean', 'median', 'std', 'count'])
print(priority_stats)

print("\n2. Resolution Time by Category:")
for category in categories.keys():
    has_cat = resolved_df[resolved_df[category] == 1]['Resolution_Days'].mean()
    no_cat = resolved_df[resolved_df[category] == 0]['Resolution_Days'].mean()
    print(f"  {category}: With={has_cat:.1f} days, Without={no_cat:.1f} days")

print("\n3. Resolution Time by Year:")
yearly_stats = resolved_df.groupby('Created_Year')['Resolution_Days'].agg(['mean', 'median', 'count'])
print(yearly_stats)

print("\n4. Correlation with text length:")
print(f"  Summary length correlation: {resolved_df['Summary_Length'].corr(resolved_df['Resolution_Days']):.3f}")
print(f"  Description length correlation: {resolved_df['Description_Length'].corr(resolved_df['Resolution_Days']):.3f}")

# Prepare features for modeling
print("\n" + "="*60)
print("PREPARING DATA FOR MODELING")
print("="*60)

feature_columns = [
    'Priority_Encoded', 'Summary_Length', 'Description_Length', 'Has_Description',
    'Created_Year', 'Created_Month', 'Created_DayOfWeek', 'Keyword_Count'
] + list(categories.keys())

X = resolved_df[feature_columns].copy()
y = resolved_df['Resolution_Days'].copy()

# Handle outliers - cap at 95th percentile
cap_value = y.quantile(0.95)
print(f"\nCapping resolution time at 95th percentile: {cap_value:.0f} days")
y_capped = y.clip(upper=cap_value)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_capped, test_size=0.2, random_state=42)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Save processed data
resolved_df.to_csv('processed_bugs.csv', index=False)
print("\n[OK] Processed data saved to 'processed_bugs.csv'")

print("\n" + "="*60)
print("FEATURE IMPORTANCE PREVIEW")
print("="*60)
print("\nFeatures to be used in modeling:")
for i, col in enumerate(feature_columns, 1):
    print(f"  {i}. {col}")

print("\n[OK] Data preparation complete!")
print("\nNext steps:")
print("  1. Run 'train_models.py' to train prediction models")
print("  2. Run 'visualize_results.py' to create charts")
