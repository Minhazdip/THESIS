import pandas as pd
import re
from collections import Counter

# Load data
df = pd.read_csv('cassandra_bugs.csv')

# Parse dates
df['Created'] = pd.to_datetime(df['Created'], format='%d/%b/%y %H:%M', errors='coerce')
df['Resolved'] = pd.to_datetime(df['Resolved'], format='%d/%b/%y %H:%M', errors='coerce')

# Calculate resolution time
df['Resolution_Days'] = (df['Resolved'] - df['Created']).dt.days

print("=" * 60)
print("CASSANDRA BUGS DATASET ANALYSIS")
print("=" * 60)

print(f"\nTotal Issues: {len(df)}")
print(f"Date Range: {df['Created'].min()} to {df['Created'].max()}")

print("\n--- RESOLUTION TIME ANALYSIS ---")
resolved_df = df[df['Resolution_Days'].notna()]
print(f"Average Resolution Time: {resolved_df['Resolution_Days'].mean():.1f} days")
print(f"Median Resolution Time: {resolved_df['Resolution_Days'].median():.1f} days")
print(f"Max Resolution Time: {resolved_df['Resolution_Days'].max():.0f} days")

print("\n--- PRIORITY vs RESOLUTION TIME ---")
priority_time = df.groupby('Priority')['Resolution_Days'].agg(['mean', 'median', 'count'])
print(priority_time)

print("\n--- KEYWORD ANALYSIS IN SUMMARIES ---")
all_text = ' '.join(df['Summary'].dropna().str.lower())
words = re.findall(r'\b[a-z]{4,}\b', all_text)
common_words = Counter(words).most_common(30)
print("Top 30 keywords:")
for word, count in common_words:
    print(f"  {word}: {count}")

print("\n--- CATEGORY DETECTION ---")
categories = {
    'Security': ['security', 'cve', 'vulnerability', 'ssl', 'tls', 'encryption'],
    'Performance': ['performance', 'slow', 'latency', 'optimization', 'memory', 'cpu'],
    'Documentation': ['doc', 'documentation', 'readme'],
    'Testing': ['test', 'testing', 'junit'],
    'Repair': ['repair', 'anticompaction'],
    'Compaction': ['compaction', 'sstable'],
    'Streaming': ['stream', 'streaming'],
    'Metrics': ['metric', 'metrics', 'monitoring'],
    'Configuration': ['config', 'configuration', 'yaml'],
}

for category, keywords in categories.items():
    pattern = '|'.join(keywords)
    count = df['Summary'].str.contains(pattern, case=False, na=False).sum()
    print(f"{category}: {count} issues")

print("\n--- YEAR-WISE TRENDS ---")
df['Year'] = df['Created'].dt.year
yearly = df.groupby('Year').size()
print(yearly)
