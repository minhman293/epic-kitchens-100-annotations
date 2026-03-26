import pandas as pd
import ast

# Load discovered patterns
patterns_df = pd.read_csv('../output/routine_named.csv')
patterns_df['pattern'] = patterns_df['pattern'].apply(ast.literal_eval)

# Load YOUR sequence file
sequence_df = pd.read_csv('../output/sequence_indexed.csv')

matches = []

for video_id in sequence_df['video_id'].unique():
    video_df = sequence_df[sequence_df['video_id'] == video_id]

    # 🔥 IMPORTANT: keep order
    video_df = video_df.sort_values(by='index')

    actions = video_df['action'].tolist()
    indices = video_df['index'].tolist()

    for i in range(len(actions) - 2):
        current = tuple(actions[i:i+3])

        for _, row in patterns_df.iterrows():
            if current == row['pattern']:
                matches.append({
                    'video_id': video_id,
                    'routine': str(row['routine_name']),
                    'start_idx': indices[i],
                    'end_idx': indices[i+2],
                    'sequence': list(current)
                })

routine_df = pd.DataFrame(matches)
routine_df.to_csv('../output/routine_matches.csv', index=False)

print("✓ Auto routines generated")