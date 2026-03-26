import pandas as pd
from collections import Counter
import ast
import matplotlib.pyplot as plt

# -----------------------------
# Utility: Safe sequence parsing
# -----------------------------
def parse_sequence(x):
    try:
        return ast.literal_eval(x)
    except:
        return []

# -----------------------------
# Step 1: Map problems → routines
# -----------------------------
def map_problems_to_routines(problems_df, routines_df):
    mapped = []

    if problems_df.empty or routines_df.empty:
        print("⚠️ Empty input to mapping")
        return pd.DataFrame(mapped)

    for _, p in problems_df.iterrows():
        for _, r in routines_df.iterrows():

            # Check same video
            if p.get('video_id') != r.get('video_id'):
                continue

            # Check index overlap
            if (
                p.get('start_idx') >= r.get('start_idx') and
                p.get('end_idx') <= r.get('end_idx')
            ):
                mapped.append({
                    'video_id': p.get('video_id'),
                    'routine': r.get('routine'),
                    'pattern': p.get('pattern', 'unknown'),
                    'duration': p.get('duration', None)
                })

    mapped_df = pd.DataFrame(mapped)

    if mapped_df.empty:
        print("⚠️ No problems mapped to any routines")

    return mapped_df


# -----------------------------
# Step 2: Analyze variation
# -----------------------------
def analyze_variation(routines_df):
    print("\n=== ROUTINE VARIATION ===")

    if routines_df.empty:
        print("⚠️ No routine data available")
        return

    grouped = routines_df.groupby('routine')

    results = []

    for name, group in grouped:
        sequences = group['sequence'].apply(tuple)
        unique = len(set(sequences))

        print(f"\n{name}")
        print(f"Total instances: {len(group)}")
        print(f"Unique variations: {unique}")

        results.append({
            'routine': name,
            'total_instances': len(group),
            'unique_variations': unique
        })

    return pd.DataFrame(results)


# -----------------------------
# Step 3: Problem distribution
# -----------------------------
def problem_stats(mapped_df):
    print("\n=== PROBLEM DISTRIBUTION ===")

    if mapped_df.empty or 'routine' not in mapped_df.columns:
        print("⚠️ No mapped problems available")
        return None

    counts = Counter(mapped_df['routine'])

    for routine, count in counts.items():
        print(f"{routine}: {count}")

    return counts


# -----------------------------
# Step 4: Visualization
# -----------------------------
def plot_problem_distribution(counts):
    if not counts:
        print("⚠️ Nothing to plot")
        return

    routines = list(counts.keys())
    values = list(counts.values())

    plt.figure()
    plt.bar(routines, values)
    plt.xticks(rotation=45)
    plt.title("Problem Distribution Across Routines")
    plt.xlabel("Routine")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig("../output/problem_distribution.png")
    plt.show()

    print("✓ Saved visualization: problem_distribution.png")


# -----------------------------
# MAIN PIPELINE
# -----------------------------
if __name__ == "__main__":

    print("=== LOADING DATA ===")

    problems_df = pd.read_csv('../output/oscillation_events.csv')
    routines_df = pd.read_csv('../output/routine_matches.csv')

    print(f"Problems: {len(problems_df)}")
    print(f"Routines: {len(routines_df)}")

    # Parse sequence safely
    if 'sequence' in routines_df.columns:
        routines_df['sequence'] = routines_df['sequence'].apply(parse_sequence)
    else:
        print("⚠️ 'sequence' column missing in routines_df")

    # -------------------------
    # Mapping
    # -------------------------
    mapped_df = map_problems_to_routines(problems_df, routines_df)
    mapped_df.to_csv('../output/problem_routine_map.csv', index=False)

    print("✓ Saved: problem_routine_map.csv")

    # -------------------------
    # Analysis
    # -------------------------
    counts = problem_stats(mapped_df)

    variation_df = analyze_variation(routines_df)
    if variation_df is not None:
        variation_df.to_csv('../output/routine_variation.csv', index=False)
        print("✓ Saved: routine_variation.csv")

    # -------------------------
    # Visualization
    # -------------------------
    plot_problem_distribution(counts)