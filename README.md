# Ozlotter Evolution Engine

This is a fully dynamic real-time prediction system for Oz Lotto using genetic evolution, entropy scoring, and historical draw simulation.

## How to Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Files
- `evolution.py`: Genetic generation
- `seed_manager.py`: Elite seed memory
- `draw_fetcher.py`: Scraping draws from lotto.net
- `simulator.py`: Division testing on predictions
- `scorer.py`: Ξ-Matrix scoring logic
- `streamlit_app.py`: UI

## Folder
- `data/`: Stores historical_draws.csv and elite_seeds.json
