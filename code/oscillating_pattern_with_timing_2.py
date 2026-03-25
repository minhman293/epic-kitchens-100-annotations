import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import Counter
import numpy as np
from pathlib import Path

# ================================
# 1. LOAD DATA
# ================================
df = pd.read_csv('../EPIC_100_train.csv')

df['action'] = df['verb'] + '(' + df['noun'] + ')'
df['start_seconds'] = pd.to_timedelta(df['start_timestamp']).dt.total_seconds()
df['stop_seconds'] = pd.to_timedelta(df['stop_timestamp']).dt.total_seconds()

# ================================
# 2. FIND OSCILLATIONS + TIMING
# ================================
def find_oscillations_with_timing(df):
    oscillations = []
    records = []

    video_ids = df['video_id'].unique()[:50]

    for video_id in video_ids:
        video_df = df[df['video_id'] == video_id].sort_values('start_seconds')

        actions = video_df['action'].tolist()
        times = video_df['start_seconds'].tolist()

        for i in range(len(actions) - 2):
            if actions[i] == actions[i+2] and actions[i] != actions[i+1]:
                a, b = actions[i], actions[i+1]
                duration = times[i+2] - times[i]

                oscillations.append((a, b))

                records.append({
                    'pattern': f"{a} ↔ {b}",
                    'action_a': a,
                    'action_b': b,
                    'duration': duration,
                    'video_id': video_id
                })

    return Counter(oscillations), pd.DataFrame(records)

osc_counts, timing_df = find_oscillations_with_timing(df)

# ================================
# 3. DATA-DRIVEN GLOBAL THRESHOLDS
# ================================
if len(timing_df) > 0:
    p50 = timing_df['duration'].quantile(0.50)
    p75 = timing_df['duration'].quantile(0.75)
    p90 = timing_df['duration'].quantile(0.90)

    print("\n📊 DATA-DRIVEN THRESHOLDS:")
    print(f"Median (P50): {p50:.1f}s")
    print(f"P75: {p75:.1f}s")
    print(f"P90: {p90:.1f}s")

    # Categorization using percentiles
    def categorize(d):
        if d < p50:
            return 'Fast (Normal)'
        elif d < p75:
            return 'Slight Delay'
        elif d < p90:
            return 'Likely Confusion'
        else:
            return 'Severe Confusion'

    timing_df['Duration_Category'] = timing_df['duration'].apply(categorize)

# ================================
# 4. PER-PATTERN BASELINE (Z-SCORE)
# ================================
pattern_stats = timing_df.groupby('pattern')['duration'].agg(['mean', 'std']).to_dict('index')

def compute_z_score(row):
    stats = pattern_stats.get(row['pattern'], None)
    if stats is None or stats['std'] == 0 or np.isnan(stats['std']):
        return 0
    return (row['duration'] - stats['mean']) / stats['std']

timing_df['z_score'] = timing_df.apply(compute_z_score, axis=1)

# ================================
# 5. OSCILLATION SCORE (NEW)
# ================================
# Combines duration + abnormality
timing_df['oscillation_score'] = (
    0.6 * (timing_df['duration'] / p90) +   # normalized duration
    0.4 * np.maximum(timing_df['z_score'], 0)  # only positive abnormality
)

# ================================
# 6. PATTERN SUMMARY
# ================================
pattern_summary = timing_df.groupby('pattern').agg({
    'duration': ['mean', 'count'],
    'oscillation_score': 'mean'
})

pattern_summary.columns = ['Avg_Duration', 'Count', 'Avg_Score']
pattern_summary = pattern_summary.sort_values('Avg_Score', ascending=False)

print("\n🚨 TOP HIGH-RISK PATTERNS (NEW SCORE):")
for pattern, row in pattern_summary.head(10).iterrows():
    print(f"{pattern}")
    print(f"  Avg Duration: {row['Avg_Duration']:.1f}s | Count: {int(row['Count'])}")
    print(f"  Score: {row['Avg_Score']:.2f}")

# ================================
# 7. VISUALIZATION UPDATE
# ================================
plt.figure(figsize=(10,6))
sns.histplot(timing_df['duration'], bins=40)

plt.axvline(p50, linestyle='--', label='P50 (Normal boundary)')
plt.axvline(p75, linestyle='--', label='P75 (Delay)')
plt.axvline(p90, linestyle='--', label='P90 (Confusion)')

plt.legend()
plt.title("Oscillation Duration Distribution (Data-Driven Thresholds)")
plt.xlabel("Duration (seconds)")
plt.ylabel("Frequency")
plt.show()

# ================================
# 8. INTERVENTION LOGIC (UPDATED)
# ================================
print("\n🤖 NEW INTERVENTION POLICY:")

print(f"""
IF oscillation_score > 1.0:
→ 🔴 HIGH PRIORITY: Immediate assistance

IF 0.5 < oscillation_score ≤ 1.0:
→ 🟠 MEDIUM PRIORITY: Monitor + assist if repeated

IF oscillation_score ≤ 0.5:
→ 🟢 NORMAL: No intervention
""")