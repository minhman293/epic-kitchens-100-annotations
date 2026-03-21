import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Load data
df = pd.read_csv('../EPIC_100_train.csv')

# Find most common verb sequences
def get_verb_sequences(df, window=3):
    sequences = []
    for video_id in df['video_id'].unique():
        video_df = df[df['video_id'] == video_id]
        verbs = video_df['verb'].tolist()
        
        # Create n-grams (sequences of length 3)
        for i in range(len(verbs) - window + 1):
            seq = tuple(verbs[i:i+window])
            sequences.append(seq)
    
    return Counter(sequences)

sequences = get_verb_sequences(df)
top_sequences = sequences.most_common(20)

# Visualize
print("Top 20 most common action sequences:")
for seq, count in top_sequences:
    print(f"{' → '.join(seq)}: {count} times")