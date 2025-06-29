
import streamlit as st
import pandas as pd
import json
from xi_matrix_engine import generate_predictions

st.set_page_config(page_title="Ozlotter-smartpro Ξ-Matrix Engine", layout="wide")

st.title("Ozlotter-smartpro Ξ-Matrix Engine")
st.markdown("Welcome to Ozlotter-smartpro — a quantum-inspired predictive engine for Oz Lotto powered by entropy shaping, multiverse simulation, and reflexive machine learning.")

# Load weights.json for dynamic control
st.subheader("Ξ Weights Configuration")
try:
    with open("weights.json", "r") as f:
        weights = json.load(f)
    st.json(weights)
except Exception as e:
    st.error(f"Failed to load weights.json: {e}")
    weights = {}

# Optional user input: upload new draw
st.subheader("Upload Recent Winning Entry (Optional)")
uploaded_file = st.file_uploader("Upload CSV (Oz Lotto draw)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Draw:")
    st.dataframe(df.head())

# Generate predictions
st.subheader("Ξ-Matrix Engine Prediction Output")
if st.button("Generate Predictions"):
    predictions = generate_predictions(70)
    st.success("Top 70 predictions ranked by Ξ-score generated.")
    results_df = pd.DataFrame([{
        "Prediction Set": " ".join(map(str, p["numbers"])),
        "Ξ-Score": round(p["xi_score"], 5)
    } for p in predictions])
    st.dataframe(results_df)

    with st.expander("🔍 Feature Breakdown (Top 5 Sets)"):
        for i, p in enumerate(predictions[:5]):
            st.markdown(f"**Set {i+1}**: `{p['numbers']}` → Ξ: `{p['xi_score']:.5f}`")
            st.json(p["features"])

st.markdown("---")
st.caption("Ozlotter-smartpro — decoding chaos, one prediction at a time.")
