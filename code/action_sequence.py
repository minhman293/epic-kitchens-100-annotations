import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

# Load data
df = pd.read_csv('../EPIC_100_train.csv')

# Create verb-noun pairs
df['action'] = df['verb'] + '(' + df['noun'] + ')'

# Get sequences of verb-noun pairs
def get_action_sequences(df, window=3):
    sequences = []
    
    for video_id in df['video_id'].unique():
        video_df = df[df['video_id'] == video_id].sort_values('start_timestamp')
        actions = video_df['action'].tolist()
        
        # Create sequences of length 'window'
        for i in range(len(actions) - window + 1):
            seq = tuple(actions[i:i+window])
            sequences.append(seq)
    
    return Counter(sequences)

# Get 3-action sequences
sequences = get_action_sequences(df, window=3)

print("\n" + "="*80)
print("TOP 30 VERB-NOUN SEQUENCES")
print("="*80)

for seq, count in sequences.most_common(30):
    print(f"{count:4} times: {' → '.join(seq)}")

top_n = 20
top_sequences = sequences.most_common(top_n)
labels = [' -> '.join(seq) for seq, _ in top_sequences]
counts = [count for _, count in top_sequences]

plt.figure(figsize=(14, 8))
plt.barh(range(len(labels)), counts, color='steelblue')
plt.yticks(range(len(labels)), labels)
plt.gca().invert_yaxis()
plt.xlabel('Frequency')
plt.ylabel('Action Sequence (window=3)')
plt.title(f'Top {top_n} Action Sequences')
plt.tight_layout()

output_dir = Path('../visualization')
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / 'action_sequence_top20.png', dpi=300, bbox_inches='tight')

plt.show()