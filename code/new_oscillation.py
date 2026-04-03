import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. LOAD AND PREPROCESS
df = pd.read_csv('../EPIC_100_train.csv')
df['action'] = df['verb'] + '(' + df['noun'] + ')'
df['start_seconds'] = pd.to_timedelta(df['start_timestamp']).dt.total_seconds()

def find_oscillations_with_timing(df):
    oscillation_events = []
    video_ids = df['video_id'].unique()[:50]
    for video_id in video_ids:
        video_df = df[df['video_id'] == video_id].sort_values('start_seconds')
        actions, times = video_df['action'].tolist(), video_df['start_seconds'].tolist()
        for i in range(len(actions) - 2):
            if actions[i] == actions[i+2] and actions[i] != actions[i+1]:
                oscillation_events.append({
                    'action_a': actions[i],
                    'pattern': f"{actions[i]} ↔ {actions[i+1]}",
                    'duration': times[i+2] - times[i]
                })
    return pd.DataFrame(oscillation_events)

timing_df = find_oscillations_with_timing(df)

# Calculate the 90th percentile as our statistical "Intervention Threshold"
threshold_90 = np.percentile(timing_df['duration'], 90)

# Confusion Rate Logic (using Action A as the primary target)
action_counts = df['action'].value_counts().to_dict()
osc_counts_by_action = timing_df['action_a'].value_counts().to_dict()

conf_list = []
for action, osc_count in osc_counts_by_action.items():
    total = action_counts.get(action, 0)
    if total > 15: # Significance filter
        conf_list.append({'Action': action, 'Freq': total, 'Osc': osc_count, 'Conf_Rate': (osc_count/total)*100})
conf_df = pd.DataFrame(conf_list).sort_values('Conf_Rate', ascending=False)

# 2. VISUALIZATION (The 2x2 Dashboard)
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
plt.suptitle("Statistical Analysis of Kitchen Disfluency: Micro-Perspective Intervention Map", fontsize=22, y=0.98)

# Q1: TOP PATTERNS (Length = Frequency, Color = Difficulty)
top_patterns = timing_df.groupby('pattern')['duration'].agg(['count', 'mean']).sort_values('count', ascending=False).head(15)
norm = plt.Normalize(top_patterns['mean'].min(), top_patterns['mean'].max())
sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=norm)

axes[0,0].barh(top_patterns.index, top_patterns['count'], color=plt.cm.RdYlGn_r(norm(top_patterns['mean'])))
axes[0,0].set_title("1. Top 15 Oscillating Patterns", fontweight='bold', pad=10)
axes[0,0].set_xlabel("Number of Occurrences (Frequency)")
axes[0,0].invert_yaxis()
# Add Colorbar to explain the colors
cbar = fig.colorbar(sm, ax=axes[0,0], fraction=0.046, pad=0.04)
cbar.set_label('Average Loop Duration (Seconds)')

# Q2: VULNERABILITY SCATTER
sns.scatterplot(data=conf_df, x='Freq', y='Conf_Rate', size='Osc', hue='Conf_Rate', palette='coolwarm', ax=axes[0,1], sizes=(50, 500))
axes[0,1].set_title("2. Action Vulnerability: Frequency vs. Confusion Rate", fontweight='bold', pad=10)
axes[0,1].set_xlabel("Total Occurrences of Action in Dataset")
axes[0,1].set_ylabel("Confusion Rate (%)")
# Label Top 3 for clarity
for i in range(min(3, len(conf_df))):
    axes[0,1].text(conf_df.iloc[i]['Freq']+30, conf_df.iloc[i]['Conf_Rate'], conf_df.iloc[i]['Action'], fontsize=10, weight='bold')

# Q3: STATISTICAL DISTRIBUTION (The "Why")
axes[1,0].hist(timing_df['duration'], bins=35, color='steelblue', alpha=0.7, edgecolor='black')
axes[1,0].axvline(threshold_90, color='red', linestyle='--', linewidth=2, label=f'90th %-ile Trigger ({threshold_90:.1f}s)')
axes[1,0].set_title("3. Global Distribution of Loop Durations", fontweight='bold', pad=10)
axes[1,0].set_xlabel("Duration of Loop (Seconds)")
axes[1,0].set_ylabel("Number of Events (Count)")
axes[1,0].legend()

# Q4: INTERVENTION PRIORITY
top_targets = conf_df.head(12)
axes[1,1].barh(top_targets['Action'], top_targets['Conf_Rate'], color='salmon')
axes[1,1].set_title("4. Intervention Priority (Highest Probability of Loop)", fontweight='bold', pad=10)
axes[1,1].set_xlabel("Confusion Rate (%)")
axes[1,1].set_ylabel("Primary Action")
axes[1,1].invert_yaxis()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('../visualization/refined_disfluency_map.png', dpi=300)
plt.show()