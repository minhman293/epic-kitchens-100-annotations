import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.patches import Patch
import ast
import os

# =========================
# 1. LOAD DATA
# =========================
routine_file = "../output/routine_matches.csv"
osc_file = "../output/oscillation_events.csv"

df = pd.read_csv(routine_file)
osc = pd.read_csv(osc_file)

# convert string list to actual list
df['actions'] = df['sequence'].apply(ast.literal_eval)

# =========================
# 2. REPETITION ANALYSIS
# =========================

def repetition_score(actions):
    return len(actions) - len(set(actions))

df['rep_score'] = df['actions'].apply(repetition_score)

functional_actions = ['stir', 'cut', 'wash', 'peel']

def classify_repetition(actions):
    verbs = [a.split('(')[0] for a in actions]

    if len(set(actions)) == 1:
        if verbs[0] in functional_actions:
            return 'functional_repetition'
        else:
            return 'problematic_repetition'

    return 'normal'

df['rep_type'] = df['actions'].apply(classify_repetition)

# =========================
# 3. SELECT SESSION (IMPORTANT)
# =========================

session_id = "P01_01"
session = df[df['video_id'] == session_id].copy()
osc_session = osc[osc['video_id'] == session_id].copy()

# =========================
# 4. OSCILLATION THRESHOLD (SESSION-LEVEL)
# =========================

mean_duration = osc_session['duration'].mean()
std_duration = osc_session['duration'].std()
threshold = mean_duration + std_duration

print(f"\nOscillation threshold (mean + std): {threshold:.2f}")

def classify_oscillation(d):
    return 'needs_assistance' if d > threshold else 'normal_adjustment'

osc_session['assist_flag'] = osc_session['duration'].apply(classify_oscillation)

# =========================
# 5. VISUALIZATION
# =========================

top_osc = osc_session.sort_values(by='duration', ascending=False).head(10)
rep_counts = session['rep_type'].value_counts()

fig, axes = plt.subplots(3, 1, figsize=(14, 14), constrained_layout=True)

# =========================
# (1) TIMELINE
# =========================

for i, (_, row) in enumerate(session.iterrows()):
    start = row['start_idx']
    end = row['end_idx']

    if row['rep_type'] == 'problematic_repetition':
        color = 'red'
    elif row['rep_type'] == 'functional_repetition':
        color = 'green'
    else:
        color = 'blue'

    axes[0].barh(y=i, width=end - start, left=start, color=color)

# overlay oscillations
for _, row in osc_session.iterrows():
    axes[0].axvspan(row['start_idx'], row['end_idx'],
                    color='orange', alpha=0.3)

axes[0].set_title(f"Session Timeline ({session_id})")
axes[0].set_xlabel("Time Index")
axes[0].set_ylabel("Action Sequence")

legend_elements = [
    Patch(facecolor='blue', label='Normal'),
    Patch(facecolor='green', label='Functional Repetition'),
    Patch(facecolor='red', label='Problematic Repetition'),
    Patch(facecolor='orange', alpha=0.3, label='Oscillation')
]

axes[0].legend(handles=legend_elements, loc='upper right')

# =========================
# (2) OSCILLATION SEVERITY
# =========================

colors = ['red' if d > threshold else 'blue' for d in top_osc['duration']]

axes[1].barh(top_osc['pattern'], top_osc['duration'], color=colors)

axes[1].axvline(threshold, color='black', linestyle='--', label='Assist Threshold')

axes[1].set_xlabel("Duration")
axes[1].set_title("Oscillation Severity (Session-level)")
axes[1].invert_yaxis()
axes[1].legend()

# =========================
# (3) REPETITION DISTRIBUTION
# =========================

axes[2].bar(rep_counts.index, rep_counts.values)

axes[2].set_title("Repetition Type Distribution (Session-level)")
axes[2].set_xlabel("Type")
axes[2].set_ylabel("Count")

# =========================
# SAVE FIGURE
# =========================

output_path = os.path.join("..", "visualization", f"analysis_pipeline_{session_id}.png")
fig.savefig(output_path, dpi=200)
print(f"Saved visualization to: {output_path}")

plt.show()

# =========================
# 6. ASSIST TRIGGER DETECTION (SESSION-BASED)
# =========================

assist_events = []

# Rule 1: problematic repetition (SESSION ONLY)
for _, row in session.iterrows():
    if row['rep_type'] == 'problematic_repetition':
        assist_events.append({
            'video_id': session_id,
            'start': row['start_idx'],
            'end': row['end_idx'],
            'type': 'repetition_issue'
        })

# Rule 2: oscillation above threshold (SESSION ONLY)
for _, row in osc_session.iterrows():
    if row['duration'] > threshold:
        assist_events.append({
            'video_id': session_id,
            'start': row['start_idx'],
            'end': row['end_idx'],
            'type': 'oscillation_issue'
        })

assist_df = pd.DataFrame(assist_events)

print("\nDetected Assist Trigger Points:")
print(assist_df.head(20))

assist_df.to_csv(f"assist_triggers_{session_id}.csv", index=False)

# =========================
# DONE
# =========================

print("\nPipeline complete.")