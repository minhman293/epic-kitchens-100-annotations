from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

df = pd.read_csv('../EPIC_100_train.csv')

# Step 1: Create sequences as strings
def get_sequence_strings(df, window=4):
    """Create sequences as text strings for clustering"""
    sequences = []
    
    for video_id in df['video_id'].unique():
        video_df = df[df['video_id'] == video_id].sort_values('start_timestamp')
        
        # Create verb sequences
        verbs = video_df['verb'].tolist()
        
        # Sliding window
        for i in range(len(verbs) - window + 1):
            seq = ' '.join(verbs[i:i+window])
            sequences.append(seq)
    
    return sequences

sequences = get_sequence_strings(df, window=4)

# Step 2: Convert sequences to vectors (one-hot encoding)
vectorizer = CountVectorizer(max_features=100)
X = vectorizer.fit_transform(sequences)

# Step 3: Cluster the sequences
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)

# Step 4: Analyze clusters
sequence_df = pd.DataFrame({
    'sequence': sequences,
    'cluster': cluster_labels
})

print("\n" + "="*80)
print("SEQUENCE CLUSTERS")
print("="*80)

for cluster in range(5):
    cluster_sequences = sequence_df[sequence_df['cluster'] == cluster]
    print(f"\n{'='*60}")
    print(f"CLUSTER {cluster} ({len(cluster_sequences)} sequences)")
    print(f"{'='*60}")
    
    # Most common sequences in this cluster
    print("\nMost common sequences:")
    seq_counts = cluster_sequences['sequence'].value_counts().head(10)
    for seq, count in seq_counts.items():
        print(f"  {count:4}x: {seq}")

# Step 5: Improved visualization (THESIS-LEVEL)

cluster_counts = sequence_df['cluster'].value_counts().sort_index()

# Step 5.1 — Rename clusters (EDIT based on your interpretation)
cluster_names = {
    0: "Alternating\n(rinse ↔ put-down)",
    1: "Interaction Loop\n(pick-up ↔ put-down)",
    2: "Cleaning Routine\n(on → rinse → off)",
    3: "Continuous Action\n(cut / stir / wash)",
    4: "Storage Action\n(open → put → close)"
}

labels = [cluster_names[i] for i in cluster_counts.index]

# Step 5.2 — Convert to proportions (more meaningful)
cluster_percent = cluster_counts / cluster_counts.sum()

# Step 5.3 — Plot
plt.figure(figsize=(12, 6))

bars = plt.bar(labels, cluster_percent.values)

plt.title('Distribution of Human Cooking Behavior Patterns', fontsize=14)
plt.xlabel('Behavior Type', fontsize=12)
plt.ylabel('Proportion of Sequences', fontsize=12)

plt.xticks(rotation=20)
plt.ylim(0, max(cluster_percent.values) * 1.2)

# Step 5.4 — Add percentage labels on bars
for bar, value in zip(bars, cluster_percent.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2%}",
        ha='center',
        va='bottom',
        fontsize=10
    )

plt.tight_layout()

# Save figure
output_dir = Path('../visualization')
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / 'behavior_pattern_distribution.png', dpi=300, bbox_inches='tight')

plt.show()