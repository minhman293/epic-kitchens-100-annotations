import pandas as pd
from collections import Counter

def extract_ngrams(sequence_df, n=3):
    patterns = []

    for video_id in sequence_df['video_id'].unique():
        video_df = sequence_df[sequence_df['video_id'] == video_id]

        # 🔥 IMPORTANT: sort by index
        video_df = video_df.sort_values(by='index')

        actions = video_df['action'].tolist()

        for i in range(len(actions) - n + 1):
            pattern = tuple(actions[i:i+n])
            patterns.append(pattern)

    return Counter(patterns)

# Load YOUR file
sequence_df = pd.read_csv('../output/sequence_indexed.csv')

# Extract patterns
ngram_counts = extract_ngrams(sequence_df, n=3)

# Filter frequent ones
frequent_patterns = [
    (pattern, count)
    for pattern, count in ngram_counts.items()
    if count >= 5   # 🔥 tune this later
]

# Save
df = pd.DataFrame(frequent_patterns, columns=['pattern', 'count'])
df = df.sort_values(by='count', ascending=False)

df.to_csv('../output/discovered_routines.csv', index=False)

print("✓ Saved discovered routines")