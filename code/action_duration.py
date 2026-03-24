import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('../EPIC_100_train.csv')

# Convert timestamps to seconds
df['start_seconds'] = pd.to_timedelta(df['start_timestamp']).dt.total_seconds()
df['stop_seconds'] = pd.to_timedelta(df['stop_timestamp']).dt.total_seconds()
df['duration'] = df['stop_seconds'] - df['start_seconds']

# Calculate gaps between consecutive actions
df = df.sort_values(['video_id', 'start_seconds'])
df['gap_to_next'] = df.groupby('video_id')['start_seconds'].diff(-1).abs()

# Analyze by action type
action_stats = df.groupby('verb').agg({
    'duration': ['mean', 'std', 'median'],
    'gap_to_next': ['mean', 'std', 'median']
}).round(2)

print("\n" + "="*80)
print("ACTION DURATION ANALYSIS")
print("="*80)
print(action_stats.sort_values(('duration', 'mean'), ascending=False).head(20))

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Duration distribution
top_verbs = df['verb'].value_counts().head(15).index
df_subset = df[df['verb'].isin(top_verbs)]

df_subset.boxplot(column='duration', by='verb', ax=ax1)
ax1.set_xlabel('Action Type')
ax1.set_ylabel('Duration (seconds)')
ax1.set_title('Action Duration Distribution')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
plt.sca(ax1)
plt.xticks(rotation=45, ha='right')

# Gap distribution
df_subset.boxplot(column='gap_to_next', by='verb', ax=ax2)
ax2.set_xlabel('Action Type')
ax2.set_ylabel('Gap to next action (seconds)')
ax2.set_title('Idle Time After Each Action')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
plt.sca(ax2)
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('../visualization/temporal_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Find actions with LONG gaps (potential intervention points)
long_gaps = df[df['gap_to_next'] > 10].groupby('verb')['gap_to_next'].agg(['count', 'mean'])
long_gaps = long_gaps[long_gaps['count'] > 20].sort_values('mean', ascending=False)

print("\n" + "="*80)
print("ACTIONS FOLLOWED BY LONG PAUSES (>10 seconds)")
print("="*80)
print("These are potential robot intervention points!")
print(long_gaps)