
import streamlit as st
import pandas as pd
from evolution import generate_generation
from seed_manager import load_seeds, save_seeds
from draw_fetcher import fetch_draws_from_lottonet, load_local_draws
from simulator import simulate_generation
from scorer import score_predictions

st.set_page_config(page_title="Ozlotter Evolution Engine", layout="wide")
st.title("🎯 Ozlotter Evolution Engine — Division 1 Builder")

# Load or fetch draws
st.sidebar.header("Historical Draws")
if st.sidebar.button("📥 Fetch Draws"):
    draws_df = fetch_draws_from_lottonet()
    st.success("Draws updated.")
else:
    draws_df = load_local_draws()

if draws_df.empty:
    st.error("No historical draws found.")
    st.stop()

# Load seeds
seeds = load_seeds()
st.sidebar.markdown("### Elite Seeds Loaded")
for s in seeds:
    st.sidebar.text(s)

# Parameters
st.sidebar.markdown("### Generation Settings")
n_children = st.sidebar.slider("How many predictions to evolve?", 50, 500, 100, step=50)
mutation_strength = st.sidebar.slider("Mutation strength (1–3)", 1, 3, 1)

mutation_pool = list(pd.Series(draws_df.values.flatten()).value_counts().head(20).index)

if st.button("🔥 Launch Evolution"):
    gen = generate_generation(seeds, mutation_pool, n_children=n_children, mutation_strength=mutation_strength)
    score_df = score_predictions(gen, draws_df)
    top_scored = score_df.head(50)

    st.subheader("🧪 Top Ξ-Scored Predictions")
    st.dataframe(top_scored[["ID", "Prediction", "XiScore"]], use_container_width=True)

    st.subheader("📊 Simulated Division Performance")
    sim_df = simulate_generation(top_scored["Prediction"].tolist(), draws_df)
    st.dataframe(sim_df, use_container_width=True)

    st.download_button("⬇️ Download Elite Predictions", top_scored.to_csv(index=False), "elite_predictions.csv")

    # Option to save best to seeds
    if st.button("💾 Save Top 10 as Elite Seeds"):
        new_elites = top_scored["Prediction"].head(10).tolist()
        save_seeds(new_elites)
        st.success("Elite seeds updated.")
else:
    st.info("Click 🔥 Launch Evolution to begin.")
