
import numpy as np
import pandas as pd
from collections import Counter

def entropy_score(number_set):
    """Calculates normalized entropy of a set"""
    counts = Counter(number_set)
    probs = [count / len(number_set) for count in counts.values()]
    return -sum(p * np.log2(p) for p in probs)

def frequency_score(number_set, frequency_map):
    """Scores set based on frequency weightings"""
    return sum(frequency_map.get(n, 0) for n in number_set)

def gap_score(number_set):
    """Measures gaps between sorted numbers (favor balanced spread)"""
    sorted_nums = sorted(number_set)
    gaps = [b - a for a, b in zip(sorted_nums[:-1], sorted_nums[1:])]
    ideal_gap = 47 / 7
    return -np.std([g - ideal_gap for g in gaps])

def get_frequency_map(historical_df):
    """Generates frequency count from historical draw data"""
    flat_numbers = historical_df.values.flatten()
    freq = dict(Counter(flat_numbers))
    return freq

def score_predictions(predictions, historical_df):
    """Scores all predictions based on entropy, frequency, and gap heuristics"""
    freq_map = get_frequency_map(historical_df)
    scored = []

    for idx, p in enumerate(predictions):
        entry = {
            "ID": f"Set-{idx+1}",
            "Prediction": p,
            "Entropy": entropy_score(p),
            "FreqScore": frequency_score(p, freq_map),
            "GapScore": gap_score(p),
        }
        # Composite Ξ-score
        entry["XiScore"] = (
            0.4 * entry["Entropy"] +
            0.4 * (entry["FreqScore"] / 100.0) +
            0.2 * entry["GapScore"]
        )
        scored.append(entry)

    df = pd.DataFrame(scored)
    return df.sort_values(by="XiScore", ascending=False)
