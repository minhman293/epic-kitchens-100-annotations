import pandas as pd
import matplotlib.pyplot as plt
import ast
import os
from matplotlib.patches import Patch

# =========================
# 1. LOAD DATA
# =========================
routine_file = "../output/routine_matches.csv"
osc_file = "../output/oscillation_events.csv"

df = pd.read_csv(routine_file)
osc = pd.read_csv(osc_file)
df['actions'] = df['sequence'].apply(ast.literal_eval)

# =========================
# 2. REPETITION ANALYSIS
# =========================
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
# 3. SELECT SESSION & CALCULATE THRESHOLD
# =========================
session_id = "P01_01"
session = df[df['video_id'] == session_id].copy()
osc_session = osc[osc['video_id'] == session_id].copy()

# Adaptive Threshold: Mean + 1 Std Dev
threshold = osc_session['duration'].mean() + osc_session['duration'].std()

# =========================
# 4. VISUALIZATION (Refined 2-Plot Layout)
# =========================
# Height ratios: 1.2 for Timeline (needs more room for sequences), 1 for Severity
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [1.2, 1]})
plt.suptitle(f"Case Study: Micro-Disfluency Analysis (Session {session_id})", fontsize=22, y=0.98, fontweight='bold')

# --- PLOT 1: SESSION TIMELINE ---
for i, (_, row) in enumerate(session.iterrows()):
    start, end = row['start_idx'], row['end_idx']
    
    # Color mapping
    color = 'blue'
    if row['rep_type'] == 'problematic_repetition': color = 'red'
    elif row['rep_type'] == 'functional_repetition': color = 'green'
    
    ax1.barh(y=i, width=end - start, left=start, color=color, alpha=0.8)

# Overlay oscillations as vertical spans
for _, row in osc_session.iterrows():
    ax1.axvspan(row['start_idx'], row['end_idx'], color='orange', alpha=0.25, label='Oscillation' if _ == 0 else "")

ax1.set_title("1. Temporal Progression & Behavior Classification", fontsize=16, pad=10)
ax1.set_xlabel("Time Index (Seconds)", fontsize=12)
ax1.set_ylabel("Action Sequence Index", fontsize=12)

legend_elements = [
    Patch(facecolor='blue', label='Normal Sequence'),
    Patch(facecolor='green', label='Functional Repetition'),
    Patch(facecolor='red', label='Problematic Repetition'),
    Patch(facecolor='orange', alpha=0.3, label='Oscillation Zone')
]
ax1.legend(handles=legend_elements, loc='upper left', fontsize=10)

# --- PLOT 2: OSCILLATION SEVERITY ---
top_osc = osc_session.sort_values(by='duration', ascending=False).head(10)
# Highlight bars that cross the threshold in red
severity_colors = ['#d62728' if d > threshold else '#1f77b4' for d in top_osc['duration']]

ax2.barh(top_osc['pattern'], top_osc['duration'], color=severity_colors)
ax2.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Assist Threshold ({threshold:.1f}s)')

ax2.set_title("2. Top 10 Bottlenecks by Duration", fontsize=16, pad=10)
ax2.set_xlabel("Duration of Loop (Seconds)", fontsize=12)
ax2.invert_yaxis()
ax2.legend(loc='lower right')

# Cleanup
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
output_path = f"../visualization/analysis_pipeline_{session_id}_refined.png"
plt.savefig(output_path, dpi=300)
plt.show()

print(f"✓ Visualization refined and saved to: {output_path}")