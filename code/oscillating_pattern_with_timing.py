import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import Counter
import numpy as np
from pathlib import Path

# 1. LOAD AND PREPROCESS DATA
df = pd.read_csv('../EPIC_100_train.csv')

# Define 'action' and timestamps IMMEDIATELY
df['action'] = df['verb'] + '(' + df['noun'] + ')'
df['start_seconds'] = pd.to_timedelta(df['start_timestamp']).dt.total_seconds()
df['stop_seconds'] = pd.to_timedelta(df['stop_timestamp']).dt.total_seconds()

def find_oscillations_with_timing(df):
    """Find cases where action A is followed by B, then A again - WITH TIMING"""
    oscillations = []
    oscillation_timings = []  # NEW: Store timing info
    video_ids = df['video_id'].unique()[:50] # Sample first 50 videos for speed
    
    for video_id in video_ids:
        video_df = df[df['video_id'] == video_id].sort_values('start_seconds')
        actions = video_df['action'].tolist()
        times = video_df['start_seconds'].tolist()  # NEW: Get timestamps
        
        for i in range(len(actions) - 2):
            # The A -> B -> A pattern
            if actions[i] == actions[i+2] and actions[i] != actions[i+1]:
                oscillations.append((actions[i], actions[i+1]))
                
                # NEW: Calculate duration of oscillation
                time_span = times[i+2] - times[i]
                oscillation_timings.append({
                    'pattern': f"{actions[i]} ↔ {actions[i+1]}",
                    'action_a': actions[i],
                    'action_b': actions[i+1],
                    'duration': time_span,
                    'video_id': video_id
                })
    
    return Counter(oscillations), pd.DataFrame(oscillation_timings)

# Run analysis - NOW RETURNS TWO THINGS
osc_counts, timing_df = find_oscillations_with_timing(df)
top_patterns = osc_counts.most_common(20)

# NEW: Analyze timing
print("\n" + "="*80)
print("OSCILLATION TIMING ANALYSIS")
print("="*80)

if len(timing_df) > 0:
    # Statistics
    avg_duration = timing_df['duration'].mean()
    median_duration = timing_df['duration'].median()
    
    # Categorize by duration
    quick_oscillations = len(timing_df[timing_df['duration'] < 10])
    medium_oscillations = len(timing_df[(timing_df['duration'] >= 10) & (timing_df['duration'] < 30)])
    slow_oscillations = len(timing_df[timing_df['duration'] >= 30])
    total = len(timing_df)
    
    print(f"Total oscillations found: {total}")
    print(f"Average duration: {avg_duration:.1f} seconds")
    print(f"Median duration: {median_duration:.1f} seconds")
    print(f"\nBreakdown by speed:")
    print(f"  Quick (<10s):  {quick_oscillations:4} ({quick_oscillations/total*100:5.1f}%) - Likely normal cooking")
    print(f"  Medium (10-30s): {medium_oscillations:4} ({medium_oscillations/total*100:5.1f}%) - Possible confusion")
    print(f"  Slow (>30s):   {slow_oscillations:4} ({slow_oscillations/total*100:5.1f}%) - Likely confusion/struggle")
    
    # Top patterns by duration
    print(f"\n{'='*80}")
    print("TOP 10 SLOWEST OSCILLATIONS (Highest confusion indicators)")
    print(f"{'='*80}")
    slowest = timing_df.nlargest(10, 'duration')
    for idx, row in slowest.iterrows():
        print(f"{row['duration']:6.1f}s: {row['pattern']}")
    
    # Average duration by pattern
    print(f"\n{'='*80}")
    print("AVERAGE DURATION BY PATTERN (Top 15)")
    print(f"{'='*80}")
    pattern_avg_duration = timing_df.groupby('pattern')['duration'].agg(['mean', 'count']).sort_values('mean', ascending=False)
    pattern_avg_duration = pattern_avg_duration[pattern_avg_duration['count'] >= 3]  # At least 3 occurrences
    for pattern, row in pattern_avg_duration.head(15).iterrows():
        print(f"{row['mean']:6.1f}s avg ({int(row['count'])}x): {pattern}")

# 2. PREPARE DATA FOR VISUALIZATION
# Create a DataFrame for patterns
pattern_df = pd.DataFrame([
    {'Pattern': f"{a} ↔ {b}", 'Count': count, 'Action_A': a, 'Action_B': b}
    for (a, b), count in top_patterns
])

# NEW: Add average duration to pattern_df
if len(timing_df) > 0:
    pattern_durations = timing_df.groupby('pattern')['duration'].mean().to_dict()
    pattern_df['Avg_Duration'] = pattern_df['Pattern'].map(pattern_durations)
    pattern_df['Duration_Category'] = pattern_df['Avg_Duration'].apply(
        lambda x: 'Quick (<10s)' if pd.notna(x) and x < 10 
        else 'Medium (10-30s)' if pd.notna(x) and x < 30 
        else 'Slow (>30s)' if pd.notna(x) 
        else 'Unknown'
    )

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

# 3. GENERATE ENHANCED DASHBOARD WITH TIMING
fig = plt.figure(figsize=(18, 12))  # Made larger to fit new plot
plt.suptitle("Kitchen Disfluency Map: Analysis of Oscillating Search Patterns", fontsize=20, y=0.98)

# Plot 1: Bar Chart of Top Patterns (NOW COLOR-CODED BY DURATION)
ax1 = plt.subplot(2, 3, 1)
if len(timing_df) > 0:
    # Color code by duration category
    colors = pattern_df['Duration_Category'].map({
        'Quick (<10s)': 'lightgreen',
        'Medium (10-30s)': 'orange',
        'Slow (>30s)': 'red',
        'Unknown': 'gray'
    })
    ax1.barh(pattern_df['Pattern'], pattern_df['Count'], color=colors)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightgreen', label='Quick (<10s) - Normal'),
        Patch(facecolor='orange', label='Medium (10-30s) - Possible confusion'),
        Patch(facecolor='red', label='Slow (>30s) - Likely confusion')
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=8)
else:
    ax1.barh(pattern_df['Pattern'], pattern_df['Count'], color='skyblue')

ax1.set_xlabel('Count', fontsize=10)
ax1.invert_yaxis()
ax1.set_title("Top 20 Oscillating Patterns\n(Color = Duration Category)", fontsize=11, fontweight='bold')
ax1.tick_params(axis='both', labelsize=8)

# Plot 2: Scatter Plot (Confusion Rate vs frequency)
ax2 = plt.subplot(2, 3, 2)
sns.scatterplot(data=conf_df, x='Frequency', y='Confusion_Rate', size='Oscillations', 
                hue='Confusion_Rate', palette='coolwarm', ax=ax2, sizes=(20, 220), legend=False)
ax2.set_title("Action Vulnerability: Frequency vs. Confusion Rate", fontsize=11, fontweight='bold')
# Annotate top 3 most confusing actions
for i in range(min(3, len(conf_df))):
    action_label = conf_df.iloc[i]['Action']
    if len(action_label) > 20:
        action_label = action_label[:17] + '...'
    ax2.text(
        conf_df.iloc[i]['Frequency'] + 10,
        conf_df.iloc[i]['Confusion_Rate'],
        action_label,
        fontsize=7,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5)
    )
ax2.tick_params(axis='both', labelsize=8)

# Plot 3: Network Graph of Relationships
ax3 = plt.subplot(2, 3, 3)
G = nx.Graph()
for (a, b), count in top_patterns[:15]:  # Use top 15 for cleaner visualization
    G.add_edge(a, b, weight=count)
pos = nx.spring_layout(G, k=0.8, iterations=50)

# Node sizes proportional to how often they appear
node_sizes = []
for node in G.nodes():
    size = sum(1 for (a, b) in top_patterns if a == node or b == node)
    node_sizes.append(size * 100)

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=node_sizes,
    node_color="skyblue",
    font_size=6,
    width=[d['weight'] / 4 for u, v, d in G.edges(data=True)],
    ax=ax3
)
ax3.set_title("Network of Behavioral Uncertainty (A ↔ B)", fontsize=11, fontweight='bold')

# Plot 4: Top Action Confusion Rates
ax4 = plt.subplot(2, 3, 4)
top_confused = conf_df.head(12)
bars = ax4.barh(top_confused['Action'], top_confused['Confusion_Rate'], color='coral')
# Highlight top 3
for i in range(min(3, len(bars))):
    bars[i].set_color('red')
ax4.set_xlabel('Confusion Rate (%)', fontsize=10)
ax4.set_title("Top 12 Actions with Highest 'Resumption Gap'", fontsize=11, fontweight='bold')
ax4.invert_yaxis()
ax4.tick_params(axis='both', labelsize=8)

# NEW Plot 5: Oscillation Duration Distribution
ax5 = plt.subplot(2, 3, 5)
if len(timing_df) > 0:
    ax5.hist(timing_df['duration'], bins=40, color='steelblue', edgecolor='black', alpha=0.7)
    ax5.axvline(x=10, color='orange', linestyle='--', linewidth=2, label='10s threshold')
    ax5.axvline(x=30, color='red', linestyle='--', linewidth=2, label='30s threshold (confusion)')
    ax5.axvline(x=timing_df['duration'].median(), color='green', linestyle='-', linewidth=2, 
                label=f'Median: {timing_df["duration"].median():.1f}s')
    ax5.set_xlabel('Oscillation Duration (seconds)', fontsize=10)
    ax5.set_ylabel('Frequency', fontsize=10)
    ax5.set_title('Distribution of Oscillation Durations\n(Longer = More likely confusion)', 
                  fontsize=11, fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.tick_params(axis='both', labelsize=8)
    
    # Add text annotation
    slow_pct = (slow_oscillations / total * 100) if total > 0 else 0
    ax5.text(0.98, 0.98, 
             f'{slow_oscillations} oscillations >30s\n({slow_pct:.1f}% likely confusion)',
             transform=ax5.transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=9)

# NEW Plot 6: Average Duration by Top Patterns
ax6 = plt.subplot(2, 3, 6)
if len(timing_df) > 0 and 'Avg_Duration' in pattern_df.columns:
    top_by_duration = pattern_df.nlargest(12, 'Avg_Duration')
    bars = ax6.barh(top_by_duration['Pattern'], top_by_duration['Avg_Duration'])
    
    # Color code bars
    for i, (idx, row) in enumerate(top_by_duration.iterrows()):
        if row['Avg_Duration'] < 10:
            bars[i].set_color('lightgreen')
        elif row['Avg_Duration'] < 30:
            bars[i].set_color('orange')
        else:
            bars[i].set_color('red')
    
    ax6.axvline(x=30, color='red', linestyle='--', alpha=0.5, label='Confusion threshold')
    ax6.set_xlabel('Average Duration (seconds)', fontsize=10)
    ax6.set_title('Slowest Oscillation Patterns\n(Top Intervention Candidates)', 
                  fontsize=11, fontweight='bold')
    ax6.invert_yaxis()
    ax6.legend(fontsize=8)
    ax6.tick_params(axis='both', labelsize=8)

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
fig.savefig('../visualization/oscillating_pattern_dashboard_with_timing.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Dashboard saved: oscillating_pattern_dashboard_with_timing.png")

# NEW: Create separate detailed timing report
if len(timing_df) > 0:
    print("\n" + "="*80)
    print("DETAILED TIMING REPORT FOR PRESENTATION")
    print("="*80)
    
    # Key insight 1: Percentage in each category
    print("\n📊 OSCILLATION SPEED BREAKDOWN:")
    print(f"   Quick actions (<10s):    {quick_oscillations:3} ({quick_oscillations/total*100:5.1f}%) → Normal cooking technique")
    print(f"   Medium duration (10-30s): {medium_oscillations:3} ({medium_oscillations/total*100:5.1f}%) → Possible confusion")
    print(f"   Slow actions (>30s):     {slow_oscillations:3} ({slow_oscillations/total*100:5.1f}%) → Clear intervention signal")
    
    # Key insight 2: Top confusion patterns
    print("\n🚨 TOP 5 INTERVENTION PRIORITIES (Slowest oscillations):")
    pattern_summary = timing_df.groupby('pattern').agg({
        'duration': ['mean', 'count']
    }).round(1)
    pattern_summary.columns = ['Avg_Duration', 'Count']
    pattern_summary = pattern_summary[pattern_summary['Count'] >= 3].sort_values('Avg_Duration', ascending=False)
    
    for i, (pattern, row) in enumerate(pattern_summary.head(5).iterrows(), 1):
        print(f"   {i}. {pattern}")
        print(f"      Average: {row['Avg_Duration']:.1f}s | Occurrences: {int(row['Count'])}")
    
    # Key insight 3: Robot intervention strategy
    print("\n🤖 ROBOT INTERVENTION STRATEGY:")
    print("   IF oscillation detected AND duration > 30s:")
    print("   → HIGH PRIORITY: Offer assistance immediately")
    print("   IF oscillation detected AND duration 10-30s:")
    print("   → MEDIUM PRIORITY: Monitor, offer help after 2nd cycle")
    print("   IF oscillation detected AND duration < 10s:")
    print("   → LOW PRIORITY: Normal behavior, no intervention")