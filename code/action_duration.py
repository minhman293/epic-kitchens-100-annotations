import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

# 1. SETUP & DATA CLEANING
df = pd.read_csv('../EPIC_100_train.csv')
df['start_seconds'] = pd.to_timedelta(df['start_timestamp']).dt.total_seconds()
df['stop_seconds'] = pd.to_timedelta(df['stop_timestamp']).dt.total_seconds()
df['duration'] = df['stop_seconds'] - df['start_seconds']

# Calculate "True Pause"
df = df.sort_values(['video_id', 'start_seconds'])
df['true_pause'] = df.groupby('video_id')['start_seconds'].shift(-1) - df['stop_seconds']
df['true_pause'] = df['true_pause'].apply(lambda x: x if x >= 0 else 0)

# 2. AGGREGATE DATA
# Most Frequent Actions
top_freq_verbs = df['verb'].value_counts().head(10).index
df_freq = df[df['verb'].isin(top_freq_verbs)]

# Longest True Pause (The "Resumption Risk")
mean_stats = df.groupby('verb')['true_pause'].mean().round(2)
top_pause_verbs = mean_stats.sort_values(ascending=False).head(10)

# 3. CREATE THE REFINED DASHBOARD (3-Plot Layout)
fig = plt.figure(figsize=(20, 14))
# Create a 2x2 grid, but we will use the bottom row as one big plot
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

plt.suptitle("Micro-Perspective Temporal Analysis: Physical Effort vs. Cognitive Struggle", 
             fontsize=24, y=0.96, fontweight='bold')

# --- PLOT 1: Top Left (Most Frequent vs Duration) ---
ax1 = fig.add_subplot(gs[0, 0])
sns.boxplot(data=df_freq, x='duration', y='verb', ax=ax1, palette='Blues', order=top_freq_verbs)
ax1.set_title("1. Physical Effort: Duration of Frequent Actions", fontsize=16, pad=15)
ax1.set_xlabel("Seconds (Execution Time)", fontsize=12)
ax1.set_ylabel("Verb", fontsize=12)

# --- PLOT 2: Top Right (Most Frequent vs True Pause) ---
ax2 = fig.add_subplot(gs[0, 1])
sns.boxplot(data=df_freq, x='true_pause', y='verb', ax=ax2, palette='Oranges', order=top_freq_verbs)
ax2.set_title("2. Cognitive Struggle: True Pause after Frequent Actions", fontsize=16, pad=15)
ax2.set_xlabel("Seconds (Idle Time)", fontsize=12)
ax2.set_ylabel("") # Remove y-label to reduce clutter

# --- PLOT 3: Bottom (Actions with Longest True Pause - Spans both columns) ---
ax3 = fig.add_subplot(gs[1, :]) # Use the entire bottom row
sns.barplot(x=top_pause_verbs.values, y=top_pause_verbs.index, ax=ax3, palette='magma')
ax3.set_title("3. Resumption Risk: Top 10 Verbs by Mean True Pause (Global)", fontsize=16, pad=15)
ax3.set_xlabel("Mean Seconds (Gap Before Next Action)", fontsize=12)
ax3.set_ylabel("Verb", fontsize=12)

# 4. FINALIZE LAYOUT
# Adjust hspace for clarity and top for suptitle margin
plt.subplots_adjust(
    left=0.1, 
    right=0.95, 
    top=0.88, 
    bottom=0.1, 
    hspace=0.4, 
    wspace=0.25
)

plt.savefig('../visualization/refined_temporal_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()