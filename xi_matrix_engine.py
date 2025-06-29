
import numpy as np
import pandas as pd
import json
import joblib
import os

# Load trained model if it exists
model_path = "ozlotpro_model.pkl"
ml_model = None
if os.path.exists(model_path):
    try:
        ml_model = joblib.load(model_path)
    except Exception:
        ml_model = None

# Load weights from JSON
def load_weights(path="weights.json"):
    with open(path, "r") as f:
        return json.load(f)

# Compute feature vector for a given set
def compute_features(number_set):
    return {
        "lambda_1": np.var(number_set),
        "lambda_2": np.std(number_set),
        "lambda_3": np.mean(number_set) / 47,
        "lambda_4": np.sum(np.diff(sorted(number_set))),
        "lambda_5": 1.0 / (1 + len(set(number_set))),
        "lambda_6": max(np.diff(sorted(number_set))),
        "lambda_7": sum(number_set) / (7 * 47),
        "lambda_8": np.random.uniform(0.2, 1.0),
        "lambda_9": np.random.uniform(0.2, 1.0),
        "lambda_10": np.std(np.diff(sorted(number_set))),
        "lambda_11": 0.0,  # will be overwritten if model available
        "lambda_12": np.random.uniform(0.2, 1.0),
        "lambda_13": np.random.uniform(0.2, 1.0),
        "lambda_14": np.random.uniform(0.2, 1.0),
        "lambda_15": np.random.uniform(0.2, 1.0)
    }

# Compute final Ξ-score
def compute_xi_score(features, weights):
    return sum(weights.get(k, 1.0) * v for k, v in features.items())

# Generate predictions
def generate_predictions(n_sets=70, weights_path="weights.json"):
    weights = load_weights(weights_path)
    predictions = []

    for _ in range(n_sets):
        number_set = sorted(np.random.choice(range(1, 48), 7, replace=False))
        features = compute_features(number_set)

        # Inject ML score if model is available
        if ml_model:
            try:
                X_pred = pd.DataFrame([list(features.values())])
                prob = ml_model.predict_proba(X_pred)[0][1]  # Probability of class 1
                features["lambda_11"] = prob
            except Exception:
                features["lambda_11"] = np.random.uniform(0.3, 1.0)
        else:
            features["lambda_11"] = np.random.uniform(0.3, 1.0)

        xi_score = compute_xi_score(features, weights)
        predictions.append({
            "numbers": number_set,
            "xi_score": round(xi_score, 5),
            "features": features
        })

    predictions = sorted(predictions, key=lambda x: x["xi_score"], reverse=True)
    return predictions
