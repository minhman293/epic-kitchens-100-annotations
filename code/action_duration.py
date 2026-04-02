import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. SETUP & DATA CLEANING
df = pd.read_csv('../EPIC_100_train.csv')
df['start_seconds'] = pd.to_timedelta(df['start_timestamp']).dt.total_seconds()
df['stop_seconds'] = pd.to_timedelta(df['stop_timestamp']).dt.total_seconds()
df['duration'] = df['stop_seconds'] - df['start_seconds']

# Calculate "True Pause" (Gap between end of action A and start of action B)
df = df.sort_values(['video_id', 'start_seconds'])
df['true_pause'] = df.groupby('video_id')['start_seconds'].shift(-1) - df['stop_seconds']
df['true_pause'] = df['true_pause'].apply(lambda x: x if x >= 0 else 0)

# 2. AGGREGATE DATA
# Most Frequent Actions
top_freq_verbs = df['verb'].value_counts().head(10).index
df_freq = df[df['verb'].isin(top_freq_verbs)]

# Longest Duration Actions (Mean)
mean_stats = df.groupby('verb').agg({'duration': 'mean', 'true_pause': 'mean'}).round(2)
top_duration_verbs = mean_stats.sort_values('duration', ascending=False).head(10)
top_pause_verbs = mean_stats.sort_values('true_pause', ascending=False).head(10)

# 3. CREATE THE 2x2 DASHBOARD
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
plt.suptitle("Micro-Perspective Temporal Analysis: Physical Effort vs. Cognitive Struggle", fontsize=22, y=0.95)

# PLOT 1: Most Frequent vs Duration
sns.boxplot(data=df_freq, x='duration', y='verb', ax=axes[0,0], palette='Blues', order=top_freq_verbs)
axes[0,0].set_title("1. Physical Effort: Duration of Most Frequent Actions", fontsize=15)
axes[0,0].set_xlabel("Seconds")

# PLOT 2: Most Frequent vs True Pause
sns.boxplot(data=df_freq, x='true_pause', y='verb', ax=axes[0,1], palette='Oranges', order=top_freq_verbs)
axes[0,1].set_title("2. Cognitive Struggle: True Pause after Most Frequent Actions", fontsize=15)
axes[0,1].set_xlabel("Seconds")

# PLOT 3: Longest Time Actions (Mean)
sns.barplot(x=top_duration_verbs['duration'], y=top_duration_verbs.index, ax=axes[1,0], palette='viridis')
axes[1,0].set_title("3. Task Complexity: Top 10 Actions by Mean Duration", fontsize=15)
axes[1,0].set_xlabel("Mean Seconds")

# PLOT 4: Actions with Longest True Pause (Mean)
sns.barplot(x=top_pause_verbs['true_pause'], y=top_pause_verbs.index, ax=axes[1,1], palette='magma')
axes[1,1].set_title("4. Resumption Risk: Top 10 Actions by Mean True Pause", fontsize=15)
axes[1,1].set_xlabel("Mean Seconds")

# Use hspace and wspace for relative spacing between subplots
# hspace=0.6 adds significant vertical space to prevent label/title overlap
plt.subplots_adjust(
    left=0.1, 
    right=0.95, 
    top=0.85, 
    bottom=0.15, 
    hspace=0.5, 
    wspace=0.3
)

# Alternatively, if you want Matplotlib to handle it automatically with specific padding
# plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=3.0)

plt.savefig('../visualization/unified_temporal_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()