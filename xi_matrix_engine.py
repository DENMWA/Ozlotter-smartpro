
import numpy as np
import pandas as pd
import json

# Load weights from JSON
def load_weights(path="weights.json"):
    with open(path, "r") as f:
        return json.load(f)

# Simulate feature extraction for a given set
def compute_features(number_set):
    # Simulated values for each component (λ₁–λ₁₅)
    return {
        "lambda_1": np.var(number_set),                      # Variance as entropy proxy
        "lambda_2": np.std(number_set),                      # Std dev for spread
        "lambda_3": np.mean(number_set) / 47,                # Mahalanobis-like normalized
        "lambda_4": np.sum(np.diff(sorted(number_set))),     # Bayesian co-pair sum
        "lambda_5": 1.0 / (1 + len(set(number_set))),        # Redundancy penalty
        "lambda_6": max(np.diff(sorted(number_set))),        # Momentum proxy
        "lambda_7": sum(number_set) / (7 * 47),              # Long-term score
        "lambda_8": np.random.uniform(0.2, 1.0),             # Mutual info (placeholder)
        "lambda_9": np.random.uniform(0.2, 1.0),             # LOF score (placeholder)
        "lambda_10": np.std(np.diff(sorted(number_set))),    # Gap diversity
        "lambda_11": np.random.uniform(0.3, 1.0),            # ML division score (placeholder)
        "lambda_12": np.random.uniform(0.2, 1.0),
        "lambda_13": np.random.uniform(0.2, 1.0),
        "lambda_14": np.random.uniform(0.2, 1.0),
        "lambda_15": np.random.uniform(0.2, 1.0)
    }

# Compute Ξ-score from weighted features
def compute_xi_score(features, weights):
    score = 0
    for key, value in features.items():
        weight = weights.get(key, 1.0)
        score += weight * value
    return score

# Generate 70 predicted sets and score them
def generate_predictions(n_sets=70, weights_path="weights.json"):
    weights = load_weights(weights_path)
    predictions = []
    for _ in range(n_sets):
        number_set = sorted(np.random.choice(range(1, 48), 7, replace=False))
        features = compute_features(number_set)
        xi_score = compute_xi_score(features, weights)
        predictions.append({
            "numbers": number_set,
            "xi_score": round(xi_score, 5),
            "features": features
        })
    # Sort by highest score
    predictions = sorted(predictions, key=lambda x: x["xi_score"], reverse=True)
    return predictions
