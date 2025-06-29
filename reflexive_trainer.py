
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from xi_matrix_engine import compute_features

# Generate feature-label dataset from user's real historical draw file
def prepare_training_data(path="Oz_Lotto_Historical_Draws.csv"):
    df = pd.read_csv(path)
    feature_rows = []
    labels = []

    for _, row in df.iterrows():
        draw_numbers = sorted([
            row["Winning Number 1"], row["Winning Number 2"], row["Winning Number 3"],
            row["Winning Number 4"], row["Winning Number 5"], row["Winning Number 6"],
            row["Winning Number 7"]
        ])
        features = compute_features(draw_numbers)
        feature_rows.append(list(features.values()))

        # Placeholder: Label = 1 if 'strong draw', simulated randomly
        labels.append(np.random.choice([0, 1], p=[0.95, 0.05]))

    X = pd.DataFrame(feature_rows, columns=features.keys())
    y = pd.Series(labels)

    return X, y

# Train model and save to disk
def train_and_save_model(out_path="ozlotpro_model.pkl"):
    X, y = prepare_training_data()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, out_path)
    return model

if __name__ == "__main__":
    model = train_and_save_model()
    print("Model trained and saved to ozlotpro_model.pkl")
