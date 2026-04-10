import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.gridspec as gridspec

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

# Calculate Confusion Rates
action_counts = df['action'].value_counts().to_dict()
osc_counts_by_action = timing_df['action_a'].value_counts().to_dict()

conf_list = []
for action, osc_count in osc_counts_by_action.items():
    total = action_counts.get(action, 0)
    if total > 15: 
        conf_list.append({'Action': action, 'Conf_Rate': (osc_count/total)*100})
conf_df = pd.DataFrame(conf_list).sort_values('Conf_Rate', ascending=False)

# 2. VISUALIZATION (3-Plot Dashboard)
fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
plt.suptitle("Statistical Analysis of Kitchen Disfluency: Logic-Driven Intervention Map", 
             fontsize=24, y=0.98, fontweight='bold')

# --- PLOT 1: Top Left (Top 15 Oscillating Patterns) ---
ax1 = fig.add_subplot(gs[0, 0])
top_patterns = timing_df.groupby('pattern')['duration'].agg(['count', 'mean']).sort_values('count', ascending=False).head(15)
norm = plt.Normalize(top_patterns['mean'].min(), top_patterns['mean'].max())
sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=norm)

ax1.barh(top_patterns.index, top_patterns['count'], color=plt.cm.RdYlGn_r(norm(top_patterns['mean'])))
ax1.set_title("1. Most Common Repetitive Patterns", fontweight='bold', fontsize=16)
ax1.set_xlabel("Number of Occurrences", fontsize=12)
ax1.invert_yaxis()
cbar = fig.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label('Avg Loop Duration (s)')

# --- PLOT 2: Top Right (Global Distribution & Threshold) ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(timing_df['duration'], bins=35, color='steelblue', alpha=0.7, edgecolor='black')
ax2.axvline(threshold_90, color='red', linestyle='--', linewidth=2, label=f'90th %-ile ({threshold_90:.1f}s)')
ax2.set_title("2. Global Loop Durations & Decision Boundary", fontweight='bold', fontsize=16)
ax2.set_xlabel("Loop Duration (Seconds)", fontsize=12)
ax2.set_ylabel("Count", fontsize=12)
ax2.legend()

# --- PLOT 3: Bottom (Intervention Priority) ---
ax3 = fig.add_subplot(gs[1, :]) # Span across both columns
top_targets = conf_df.head(12)
ax3.barh(top_targets['Action'], top_targets['Conf_Rate'], color='salmon')
ax3.set_title("3. Actions with Highest Confusion Probability", fontweight='bold', fontsize=16)
ax3.set_xlabel("Confusion Rate (%)", fontsize=12)
ax3.set_ylabel("Primary Action", fontsize=12)
ax3.invert_yaxis()

# Final Polish
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('../visualization/refined_disfluency_map_v2.png', dpi=300)
plt.show()