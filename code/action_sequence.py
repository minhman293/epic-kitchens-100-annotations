import pandas as pd
from collections import Counter
from pathlib import Path

# Load data
df = pd.read_csv('../EPIC_100_train.csv')

# Create verb-noun pairs
df['action'] = df['verb'] + '(' + df['noun'] + ')'
df['start_seconds'] = pd.to_timedelta(df['start_timestamp']).dt.total_seconds()

def get_action_sequences(df):
    """Return structured sequences per video"""
    all_sequences = []

    for video_id in df['video_id'].unique():
        video_df = df[df['video_id'] == video_id].sort_values('start_seconds')

        actions = video_df['action'].tolist()
        times = video_df['start_seconds'].tolist()

        for i in range(len(actions)):
            all_sequences.append({
                'video_id': video_id,
                'index': i,
                'action': actions[i],
                'timestamp': times[i]
            })

    return pd.DataFrame(all_sequences)

def get_window_sequences(df, window=3):
    """Keep your original functionality (for frequency analysis)"""
    sequences = []

    for video_id in df['video_id'].unique():
        video_df = df[df['video_id'] == video_id].sort_values('start_seconds')
        actions = video_df['action'].tolist()

        for i in range(len(actions) - window + 1):
            seq = tuple(actions[i:i+window])
            sequences.append(seq)

    return Counter(sequences)

# Run
sequence_df = get_action_sequences(df)
sequence_df.to_csv('../output/sequence_indexed.csv', index=False)

print("✓ Saved indexed sequences")

# Optional: keep your original analysis
sequences = get_window_sequences(df, window=3)

print("\nTOP 20 SEQUENCES:")
for seq, count in sequences.most_common(20):
    print(count, ":", " → ".join(seq))