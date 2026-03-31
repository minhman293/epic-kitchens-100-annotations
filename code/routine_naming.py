import pandas as pd
import ast

df = pd.read_csv('../output/discovered_routines.csv')
df['pattern'] = df['pattern'].apply(ast.literal_eval)

def name_routine(pattern):
    p = " ".join(pattern)

    # Washing operations
    if "wash" in p or "rinse" in p:
        return "washing"

    # Cutting operations (including slicing, dicing, chopping, trimming)
    elif "cut" in p or "slice" in p or "dice" in p or "trim" in p or "top(" in p:
        return "cutting"

    # Chopping operations
    elif "chop" in p:
        return "chopping"

    # Stirring operations
    elif "stir" in p:
        return "stirring"

    # Peeling operations
    elif "peel" in p or "skin(" in p or ("take-off" in p and "skin" in p) or ("remove" in p and "skin" in p):
        return "peeling"

    # Flipping operations
    elif "flip" in p:
        return "flipping"

    # Pouring and dispensing operations
    elif "pour" in p:
        return "pouring"

    # Folding, stretching, kneading, and dough shaping operations
    elif "fold" in p or "stretch" in p or "knead" in p or "shape" in p or "squeeze" in p or "roll" in p:
        return "folding_shaping"

    # Pressing, operating, sharpening
    elif "press" in p or "sharpen" in p or "tap(" in p:
        return "operation"

    # Cleaning and wiping surfaces
    elif "wipe" in p or "clean" in p or "scour" in p or "spray" in p:
        return "cleaning"

    # Drying, shaking, wringing operations
    elif "shake" in p or "dry" in p or "wring" in p or "unfold" in p or "tear" in p:
        return "drying"

    # Breaking up, deboning, shredding, separating
    elif "break-up" in p or "debone" in p or "shred" in p or "discard" in p or "remove-from" in p:
        return "separating"

    # Container/drawer operations (open and close together)
    elif "open" in p and "close" in p:
        return "container_usage"

    # General object transfer and manipulation
    elif "put" in p or "pick-up" in p or "place" in p or "get(" in p or "take" in p or "move" in p:
        return "object_transfer"

    else:
        return "other"

df['routine_name'] = df['pattern'].apply(name_routine)

df.to_csv('../output/routine_named.csv', index=False)

print("✓ Saved named routines")