
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

mean_hilbert_vector = np.array([0.1224046523577074, 0.2449260807623146, 0.37158138122708256, 0.49183735432188114, 0.6141252306326913, 0.7373473153186775, 0.8656374804400092])

def hilbert_similarity(vec, reference):
    v1 = np.array(vec) / 47.0
    v2 = reference
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def compute_features(number_set):
    gaps = np.diff(sorted(number_set))
    return {
        "lambda_1": np.var(number_set),
        "lambda_2": np.std(number_set),
        "lambda_3": np.mean(number_set) / 47,
        "lambda_4": np.sum(gaps),
        "lambda_5": 1.0 / (1 + len(set(number_set))),
        "lambda_6": max(gaps),
        "lambda_7": sum(number_set) / (7 * 47),
        "lambda_8": np.random.uniform(0.2, 1.0),
        "lambda_9": np.random.uniform(0.2, 1.0),
        "lambda_10": np.std(gaps),
        "lambda_11": np.random.uniform(0.3, 1.0),
        "lambda_12": np.random.uniform(0.2, 1.0),
        "lambda_13": np.random.uniform(0.2, 1.0),
        "lambda_14": np.random.uniform(0.2, 1.0),
        "lambda_15": np.random.uniform(0.2, 1.0),
        "lambda_16": hilbert_similarity(number_set, mean_hilbert_vector)
    }

def prepare_training_data(path="Oz_Lotto_Historical_Draws.csv"):
    df = pd.read_csv(path)
    feature_rows = []
    labels = []

    for _, row in df.iterrows():
        try:
            draw_numbers = sorted([
                row["Winning Number 1"], row["Winning Number 2"], row["Winning Number 3"],
                row["Winning Number 4"], row["Winning Number 5"], row["Winning Number 6"],
                row["Winning Number 7"]
            ])
            features = compute_features(draw_numbers)
            feature_rows.append(list(features.values()))
            labels.append(np.random.choice([0, 1], p=[0.95, 0.05]))  # Simulated label
        except:
            continue

    X = pd.DataFrame(feature_rows, columns=features.keys())
    y = pd.Series(labels)
    return X, y

def train_and_save_model(out_path="ozlotpro_model.pkl"):
    X, y = prepare_training_data()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, out_path)
    return model

if __name__ == "__main__":
    model = train_and_save_model()
    print("Model trained and saved to ozlotpro_model.pkl")
