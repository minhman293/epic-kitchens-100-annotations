import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. SETUP & DATA CLEANING
df = pd.read_csv('../EPIC_100_train.csv')
df['action'] = df['verb'] + "(" + df['noun'] + ")"
df['start_seconds'] = pd.to_timedelta(df['start_timestamp']).dt.total_seconds()
df['stop_seconds'] = pd.to_timedelta(df['stop_timestamp']).dt.total_seconds()

# Calculate "True Pause"
df = df.sort_values(['video_id', 'start_seconds'])
df['true_pause'] = df.groupby('video_id')['start_seconds'].shift(-1) - df['stop_seconds']
df['true_pause'] = df['true_pause'].apply(lambda x: x if x >= 0 else 0)

# 2. FIND TOP HIGH-PAUSE ACTIONS (VERB-NOUN)
# We filter for actions that occur at least 10 times to ensure statistical significance
action_counts = df['action'].value_counts()
significant_actions = action_counts[action_counts >= 10].index
df_sig = df[df['action'].isin(significant_actions)]

# Calculate Mean Pause and select Top 15 "Micro-Bottlenecks"
top_pause_actions = df_sig.groupby('action')['true_pause'].mean().sort_values(ascending=False).head(15).index
df_plot = df_sig[df_sig['action'].isin(top_pause_actions)]

# 3. VISUALIZATION WITH THRESHOLDS
plt.figure(figsize=(14, 10))
colors = sns.color_palette("rocket_r", n_colors=len(top_pause_actions))

# Draw the Boxplot
ax = sns.boxplot(
    data=df_plot, 
    x='true_pause', 
    y='action', 
    order=top_pause_actions,
    palette=colors,
    showfliers=True,  # Show the "Struggle" outliers
    fliersize=4,
    linewidth=1.5
)

# 4. CALCULATE AND DRAW INTERVENTION THRESHOLDS
for i, action in enumerate(top_pause_actions):
    data = df[df['action'] == action]['true_pause'].dropna()
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    threshold = q3 + (1.5 * iqr)
    
    # Draw a dashed red line for the "Robot Intervention Trigger"
    plt.vlines(x=threshold, ymin=i-0.4, ymax=i+0.4, color='red', linestyle='--', alpha=0.7, label='Intervention Trigger' if i == 0 else "")
    
    # Highlight the "Struggle Zone"
    plt.axvspan(threshold, df_plot['true_pause'].max(), ymin=1 - (i+1)/15, ymax=1 - i/15, color='red', alpha=0.05)

# Formatting
plt.title("Action-Specific Intervention Thresholds: High-Struggle Verb-Noun Pairs", fontsize=18, pad=20)
plt.xlabel("True Pause Duration (Seconds) \n [Dashed Line = Statistical Threshold for Robot Intervention]", fontsize=12)
plt.ylabel("Micro-Action (Verb + Noun)", fontsize=12)
plt.grid(axis='x', linestyle=':', alpha=0.4)
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig('../visualization/intervention_thresholds.png', dpi=300)
plt.show()

# 5. PRINT THE SUMMARY FOR THE REPORT
print("\n" + "="*60)
print("ROBOT INTERVENTION LOGIC (MICRO-PERSPECTIVE)")
print("="*60)
for action in top_pause_actions:
    data = df[df['action'] == action]['true_pause'].dropna()
    threshold = data.quantile(0.75) + (1.5 * (data.quantile(0.75) - data.quantile(0.25)))
    print(f"Action: {action:30} Trigger Assistance after: {threshold:5.2f}s")