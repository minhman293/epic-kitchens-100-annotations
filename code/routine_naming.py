import pandas as pd
import ast

df = pd.read_csv('../output/discovered_routines.csv')
df['pattern'] = df['pattern'].apply(ast.literal_eval)

def name_routine(pattern):
    p = " ".join(pattern)

    if "wash" in p or "rinse" in p:
        return "washing"

    elif "cut" in p:
        return "cutting"

    elif "stir" in p:
        return "stirring"

    elif "open" in p and "close" in p:
        return "container_usage"

    elif "put" in p:
        return "object_transfer"

    else:
        return "other"

df['routine_name'] = df['pattern'].apply(name_routine)

df.to_csv('../output/routine_named.csv', index=False)

print("✓ Saved named routines")