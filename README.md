# Ozlotter Evolution Engine

This is a fully dynamic real-time prediction system for Oz Lotto using genetic evolution, entropy scoring, and historical draw simulation.

## How to Run

### Original Version
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Enhanced Version (with AI optimization)
```bash
pip install -r enhanced_requirements.txt
streamlit run enhanced_streamlit_app.py
```

## Files
- `evolution.py`: Genetic generation
- `seed_manager.py`: Elite seed memory
- `draw_fetcher.py`: Scraping draws from lotto.net
- `simulator.py`: Division testing on predictions
- `scorer.py`: Ξ-Matrix scoring logic
- `streamlit_app.py`: Original UI
- `enhanced_streamlit_app.py`: Enhanced UI with 10 AI optimization methods

## Enhanced Features
- Neural Networks (LSTM) for sequence prediction
- Advanced genetic algorithms with adaptive mutation
- Multi-dimensional pattern analysis
- Chaos theory and fractal analysis
- Cross-lottery intelligence
- Market psychology analysis
- Dynamic weight optimization
- Real-time adaptive learning
- Enhanced scoring system
- Professional analytics dashboard

## Folder
- `data/`: Stores historical_draws.csv and elite_seeds.json
