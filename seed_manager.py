
import json
import os

SEED_FILE = "data/elite_seeds.json"

def load_seeds():
    """Loads elite seeds from JSON file"""
    if not os.path.exists(SEED_FILE):
        return []
    with open(SEED_FILE, "r") as f:
        seeds = json.load(f)
    return [s for s in seeds if len(s) == 7 and all(1 <= int(x) <= 47 for x in s)]

def save_seeds(seed_sets, top_n=20):
    """Saves top elite seeds to JSON (prunes to top N by default)"""
    existing = load_seeds()
    combined = existing + seed_sets
    unique_seeds = []
    seen = set()
    for seed in combined:
        key = tuple(sorted(seed))
        if key not in seen:
            unique_seeds.append(seed)
            seen.add(key)
    elite = unique_seeds[:top_n]
    os.makedirs("data", exist_ok=True)
    with open(SEED_FILE, "w") as f:
        json.dump(elite, f, indent=2)
    return elite
