
import streamlit as st
import pandas as pd
import json
import os
import joblib
from xi_matrix_engine import generate_predictions
from reflexive_trainer import train_and_save_model

st.set_page_config(page_title="Ozlotter-smartpro Ξ-Matrix Engine", layout="wide")
st.title("Ozlotter-smartpro Ξ-Matrix Engine")

tab1, tab2 = st.tabs(["🔮 Predictions", "🧠 ML Engine & Retraining"])

# ===========================
# 🔮 Prediction Tab
# ===========================
with tab1:
    st.subheader("Ξ Weights Configuration")
    try:
        with open("weights.json", "r") as f:
            weights = json.load(f)
        st.json(weights)
    except Exception as e:
        st.error(f"Failed to load weights.json: {e}")

    st.subheader("Ξ-Matrix Predictions")
    if st.button("Generate Predictions"):
        predictions = generate_predictions(70)
        st.success("Top 70 predictions ranked by Ξ-score.")
        results_df = pd.DataFrame([{
            "Prediction Set": " ".join(map(str, p["numbers"])),
            "Ξ-Score": round(p["xi_score"], 5)
        } for p in predictions])
        st.dataframe(results_df)

        with st.expander("🔍 Feature Breakdown (Top 5 Sets)"):
            for i, p in enumerate(predictions[:5]):
                st.markdown(f"**Set {i+1}**: `{p['numbers']}` → Ξ: `{p['xi_score']:.5f}`")
                st.json(p["features"])

# ===========================
# 🧠 ML Tab
# ===========================
with tab2:
    st.subheader("Upload New Winning Entry")
    uploaded = st.file_uploader("Upload a new winning draw (CSV)", type=["csv"])

    if uploaded:
        new_draw = pd.read_csv(uploaded)
        st.write("Uploaded Draw:")
        st.dataframe(new_draw)

        # Append to historical draw file
        hist_path = "Oz_Lotto_Historical_Draws.csv"
        if os.path.exists(hist_path):
            existing = pd.read_csv(hist_path)
            combined = pd.concat([existing, new_draw], ignore_index=True)
            combined.drop_duplicates(inplace=True)
            combined.to_csv(hist_path, index=False)
            st.success("New draw appended to historical record.")
        else:
            new_draw.to_csv(hist_path, index=False)
            st.warning("Historical record not found. Created new one.")

    # Retrain model
    if st.button("♻️ Retrain ML Model"):
        model = train_and_save_model()
        st.success("Model retrained and saved to ozlotpro_model.pkl")

        # Confirm updated size
        updated = pd.read_csv("Oz_Lotto_Historical_Draws.csv")
        st.info(f"Historical draw count: {len(updated)} rows")

st.markdown("---")
st.caption("Ozlotter-smartpro — Reflexive, Quantum-Aware, and Always Evolving.")
