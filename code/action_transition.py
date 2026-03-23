import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from pathlib import Path

df = pd.read_csv('../EPIC_100_train.csv')

# Sort properly
df = df.sort_values(['video_id', 'start_timestamp'])

# Create action token
df['action'] = df['verb'] + "_" + df['noun']

# Group into sequences
sequences = df.groupby('video_id')['action'].apply(list)

print(sequences.iloc[0][:20])  # preview

transitions = []

for seq in sequences:
    for i in range(len(seq) - 1):
        transitions.append((seq[i], seq[i+1]))

transition_counts = Counter(transitions)

print("\nTop transitions:")
for t, c in transition_counts.most_common(20):
    print(t, c)

transition_df = pd.DataFrame(transitions, columns=['from', 'to'])

top_transitions = transition_df.value_counts().head(20).reset_index()
top_transitions.columns = ['from', 'to', 'count']

print(top_transitions)

# Keep only frequent transitions
min_count = 50
filtered_transitions = [(f, t) for (f, t), c in transition_counts.items() if c >= min_count]

transition_df = pd.DataFrame(filtered_transitions, columns=['from', 'to'])

# Use verbs only (higher-level abstraction)
df['verb_only'] = df['verb']

sequences_verb = df.groupby('video_id')['verb_only'].apply(list)

transitions_verb = []

for seq in sequences_verb:
    for i in range(len(seq) - 1):
        transitions_verb.append((seq[i], seq[i+1]))

transition_counts_verb = Counter(transitions_verb)

print("\nTop VERB transitions:")
for t, c in transition_counts_verb.most_common(20):
    print(t, c)

import seaborn as sns
import matplotlib.pyplot as plt

transition_df_verb = pd.DataFrame(transitions_verb, columns=['from', 'to'])

matrix = pd.crosstab(
    transition_df_verb['from'],
    transition_df_verb['to']
)

# Focus on top verbs only (avoid huge matrix)
top_verbs = df['verb'].value_counts().head(15).index
matrix = matrix.loc[top_verbs, top_verbs]

plt.figure(figsize=(10, 8))
sns.heatmap(matrix, cmap='Blues')
plt.title("Verb Transition Patterns")
plt.xlabel("Next Action")
plt.ylabel("Current Action")

output_dir = Path('../visualization')
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / 'verb_transition_patterns.png', dpi=300, bbox_inches='tight')

plt.show()