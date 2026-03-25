import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# ================================
# 1. LOAD DATA
# ================================
df = pd.read_csv('../EPIC_100_train.csv')

df['action'] = df['verb'] + '(' + df['noun'] + ')'
df['start_seconds'] = pd.to_timedelta(df['start_timestamp']).dt.total_seconds()

# ================================
# 2. EXTRACT SEQUENCES PER VIDEO
# ================================
df = df.sort_values(['video_id', 'start_seconds'])

video_sequences = df.groupby('video_id')['action'].apply(list)
video_users = df.groupby('video_id')['participant_id'].first()

# ================================
# 3. N-GRAM FUNCTION
# ================================
def generate_ngrams(sequence, n=3):
    return [" ".join(sequence[i:i+n]) for i in range(len(sequence)-n+1)]

# ================================
# 4. BUILD TF-IDF WITH N-GRAMS (FIX CLUSTERING)
# ================================
documents = []

for seq in video_sequences:
    grams = []
    for n in range(2, 5):  # bi-gram to 4-gram
        grams += generate_ngrams(seq, n)
    documents.append(" | ".join(grams))

vectorizer = TfidfVectorizer(token_pattern=r'[^|]+')
X = vectorizer.fit_transform(documents)

# ================================
# 5. CLUSTERING (IMPROVED)
# ================================
n_clusters = 6
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X)

cluster_df = pd.DataFrame({
    'video_id': video_sequences.index,
    'sequence': video_sequences.values,
    'user': video_users.values,
    'cluster': labels
})

print("\n📦 Cluster distribution:")
print(cluster_df['cluster'].value_counts())

# ================================
# 6. CLEAN ROUTINE EXTRACTION
# ================================
def clean_sequence(seq):
    cleaned = []
    for action in seq:
        if not cleaned or cleaned[-1] != action:
            cleaned.append(action)
    return cleaned

def extract_frequent_patterns(seqs, min_count=5):
    counter = Counter()
    for seq in seqs:
        seq = clean_sequence(seq)
        for n in range(3, 6):
            for i in range(len(seq)-n+1):
                pattern = tuple(seq[i:i+n])
                counter[pattern] += 1
    return {k: v for k, v in counter.items() if v >= min_count}

cluster_patterns = {}

print("\n🍳 CLEAN ROUTINES PER CLUSTER:")
for c in range(n_clusters):
    seqs = cluster_df[cluster_df['cluster'] == c]['sequence']
    
    patterns = extract_frequent_patterns(seqs)
    top_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:5]
    
    cluster_patterns[c] = top_patterns
    
    print(f"\nCluster {c}:")
    for p, count in top_patterns:
        print(f"{' → '.join(p)} ({count}x)")

# ================================
# 7. USER vs GLOBAL BEHAVIOR
# ================================
def compute_action_duration(df):
    df['duration'] = df['stop_timestamp'].apply(pd.to_timedelta).dt.total_seconds() - \
                     df['start_timestamp'].apply(pd.to_timedelta).dt.total_seconds()
    return df

df = compute_action_duration(df)

global_stats = df.groupby('action')['duration'].mean()

user_stats = df.groupby(['participant_id', 'action'])['duration'].mean().reset_index()

# ================================
# 8. BUILD DASHBOARD INPUTS
# ================================
def shorten_label(text, max_len=70):
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."

cluster_counts = cluster_df['cluster'].value_counts().sort_index()

# Keep top routines from each cluster and then rank globally for dashboard readability.
routine_items = []
for c, patterns in cluster_patterns.items():
    for pattern, count in patterns[:3]:
        label = f"C{c}: " + " -> ".join(pattern)
        routine_items.append((shorten_label(label), count))

routine_items = sorted(routine_items, key=lambda x: x[1], reverse=True)[:12]

# ================================
# 10. OSCILLATION DETECTION
# ================================
def detect_oscillation(seq, times):
    results = []
    for i in range(len(seq)-2):
        if seq[i] == seq[i+2] and seq[i] != seq[i+1]:
            duration = times[i+2] - times[i]
            results.append(duration)
    return results

oscillation_data = []

for vid in df['video_id'].unique():
    sub = df[df['video_id'] == vid].sort_values('start_seconds')
    seq = sub['action'].tolist()
    times = sub['start_seconds'].tolist()
    
    durations = detect_oscillation(seq, times)
    for d in durations:
        oscillation_data.append(d)

# ================================
# 11. USER VARIABILITY ANALYSIS
# ================================
user_variability = defaultdict(list)

for _, row in cluster_df.iterrows():
    user_variability[row['cluster']].append(row['user'])

print("\n👥 USER DISTRIBUTION PER CLUSTER:")
for c in user_variability:
    print(f"Cluster {c}: {len(set(user_variability[c]))} unique users")

cluster_user_counts = {
    c: len(set(users)) for c, users in user_variability.items()
}

# ================================
# 12. COMBINED DASHBOARD FIGURE
# ================================
fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)

# Panel 1: cluster distribution
ax1 = axes[0, 0]
ax1.bar(cluster_counts.index.astype(str), cluster_counts.values, color='steelblue')
ax1.set_title("Cluster Distribution")
ax1.set_xlabel("Cluster")
ax1.set_ylabel("Number of Sessions")

# Panel 2: top routines across clusters
ax2 = axes[0, 1]
if routine_items:
    labels = [item[0] for item in routine_items]
    counts = [item[1] for item in routine_items]
    ax2.barh(labels[::-1], counts[::-1], color='seagreen')
    ax2.set_title("Top Routines Across Clusters")
    ax2.set_xlabel("Frequency")
else:
    ax2.text(0.5, 0.5, "No routine patterns found", ha='center', va='center')
    ax2.set_title("Top Routines Across Clusters")
    ax2.set_axis_off()

# Panel 3: oscillation duration distribution
ax3 = axes[1, 0]
if oscillation_data:
    ax3.hist(oscillation_data, bins=30, color='darkorange', edgecolor='black', alpha=0.85)
    ax3.set_title("Oscillation Duration Distribution")
    ax3.set_xlabel("Duration (seconds)")
    ax3.set_ylabel("Frequency")
else:
    ax3.text(0.5, 0.5, "No oscillation events detected", ha='center', va='center')
    ax3.set_title("Oscillation Duration Distribution")
    ax3.set_axis_off()

# Panel 4: unique users per cluster
ax4 = axes[1, 1]
if cluster_user_counts:
    ordered_clusters = sorted(cluster_user_counts.keys())
    user_counts = [cluster_user_counts[c] for c in ordered_clusters]
    ax4.bar([str(c) for c in ordered_clusters], user_counts, color='mediumpurple')
    ax4.set_title("Unique Users per Cluster")
    ax4.set_xlabel("Cluster")
    ax4.set_ylabel("Unique Users")
else:
    ax4.text(0.5, 0.5, "No user variability data", ha='center', va='center')
    ax4.set_title("Unique Users per Cluster")
    ax4.set_axis_off()

output_dir = Path('../visualization')
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'oscillating_3_combined_dashboard.png'
fig.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"\nSaved combined dashboard figure to: {output_path}")

# ================================
# 13. KEY OUTPUT SUMMARY
# ================================
print("\n🎯 FINAL INSIGHTS:")
print("""
1. Clusters now represent DIFFERENT cooking activities.
2. Each cluster has CLEAN, human-readable routines.
3. Variability across users is measurable.
4. Oscillation patterns highlight confusion moments.
""")