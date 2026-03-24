import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import Counter
import numpy as np
from pathlib import Path

# 1. LOAD AND PREPROCESS DATA
df = pd.read_csv('../EPIC_100_train.csv')

# Define 'action' IMMEDIATELY so it's available for the function
df['action'] = df['verb'] + '(' + df['noun'] + ')'
df['start_seconds'] = pd.to_timedelta(df['start_timestamp']).dt.total_seconds()

def find_oscillations(df):
    """Find cases where action A is followed by B, then A again"""
    oscillations = []
    video_ids = df['video_id'].unique()[:50] # Sample first 50 videos for speed
    
    for video_id in video_ids:
        video_df = df[df['video_id'] == video_id].sort_values('start_seconds')
        actions = video_df['action'].tolist()
        
        for i in range(len(actions) - 2):
            # The A -> B -> A pattern
            if actions[i] == actions[i+2] and actions[i] != actions[i+1]:
                oscillations.append((actions[i], actions[i+1]))
    
    return Counter(oscillations)

# Run analysis
osc_counts = find_oscillations(df)
top_patterns = osc_counts.most_common(20)

# 2. PREPARE DATA FOR VISUALIZATION
# Create a DataFrame for patterns
pattern_df = pd.DataFrame([
    {'Pattern': f"{a} ↔ {b}", 'Count': count, 'Action_A': a, 'Action_B': b}
    for (a, b), count in top_patterns
])

# Calculate Confusion Rates
action_osc_totals = {}
for (a, b), count in osc_counts.items():
    action_osc_totals[a] = action_osc_totals.get(a, 0) + count

conf_data = []
for action, osc_count in action_osc_totals.items():
    total_occurrences = len(df[df['action'] == action])
    if total_occurrences > 20: # Filter out rare actions
        conf_data.append({
            'Action': action,
            'Oscillations': osc_count,
            'Frequency': total_occurrences,
            'Confusion_Rate': (osc_count / total_occurrences) * 100
        })
conf_df = pd.DataFrame(conf_data).sort_values('Confusion_Rate', ascending=False)

# 3. GENERATE DASHBOARD
fig = plt.figure(figsize=(14, 10))
plt.suptitle("Kitchen Disfluency Map: Analysis of Oscillating Search Patterns", fontsize=18, y=0.98)

# Plot 1: Bar Chart of Top Patterns
ax1 = plt.subplot(2, 2, 1)
sns.barplot(data=pattern_df, x='Count', y='Pattern', palette='viridis', ax=ax1)
ax1.set_title("Top 20 Oscillating Patterns (Searching Behavior)", fontsize=12)

# Plot 2: Scatter Plot (Confusion Rate vs frequency)
ax2 = plt.subplot(2, 2, 2)
sns.scatterplot(data=conf_df, x='Frequency', y='Confusion_Rate', size='Oscillations', hue='Confusion_Rate', palette='coolwarm', ax=ax2, sizes=(20, 220))
ax2.set_title("Action Vulnerability: Frequency vs. Confusion Rate", fontsize=12)
# Annotate top 5 most confusing actions
for i in range(min(5, len(conf_df))):
    ax2.text(
        conf_df.iloc[i]['Frequency'] + 5,
        conf_df.iloc[i]['Confusion_Rate'],
        conf_df.iloc[i]['Action'],
        fontsize=8
    )

# Plot 3: Network Graph of Relationships
ax3 = plt.subplot(2, 2, 3)
G = nx.Graph()
for (a, b), count in top_patterns:
    G.add_edge(a, b, weight=count)
pos = nx.spring_layout(G, k=0.5)
nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=1100,
    node_color="skyblue",
    font_size=7,
    width=[d['weight'] / 6 for u, v, d in G.edges(data=True)],
    ax=ax3
)
ax3.set_title("Network of Behavioral Uncertainty (A ↔ B)", fontsize=12)

# Plot 4: Top Action Confusion Rates
ax4 = plt.subplot(2, 2, 4)
sns.barplot(data=conf_df.head(15), x='Confusion_Rate', y='Action', palette='magma', ax=ax4)
ax4.set_title("Top 15 Actions with Highest 'Resumption Gap'", fontsize=12)

for ax in [ax1, ax2, ax3, ax4]:
    ax.tick_params(axis='both', labelsize=8)

plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=2.2, w_pad=2.0)
fig.savefig('../visualization/oscillating_pattern_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()