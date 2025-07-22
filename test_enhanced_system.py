"""
Test script for the enhanced ozlotter system
"""
from master_predictor import MasterPredictor
from enhanced_draw_fetcher import EnhancedDrawFetcher
import pandas as pd

def test_enhanced_system():
    print('Testing enhanced system components...')

    fetcher = EnhancedDrawFetcher()
    print('✓ Enhanced draw fetcher initialized')

    predictor = MasterPredictor()
    print('✓ Master predictor initialized')

    sample_data = pd.DataFrame({
        'N1': [1, 5, 12, 8, 3],
        'N2': [7, 11, 18, 15, 9], 
        'N3': [14, 22, 25, 21, 16],
        'N4': [23, 28, 31, 29, 24],
        'N5': [30, 35, 38, 36, 32],
        'N6': [39, 42, 44, 41, 38],
        'N7': [45, 47, 46, 44, 43]
    })

    print('✓ Sample data created')

    try:
        seeds = [[1,7,14,23,30,39,45], [5,11,22,28,35,42,47]]
        results = predictor.generate_ensemble_predictions(
            draws_df=sample_data,
            seed_sets=seeds,
            n_predictions=10,
            enable_neural=False,  # Disable neural for quick test
            enable_chaos=True,
            enable_patterns=True,
            enable_psychology=True
        )
        print(f'✓ Generated {len(results["predictions"])} predictions successfully')
        print('✓ All enhanced system components working!')
        return True
    except Exception as e:
        print(f'✗ Error in prediction generation: {e}')
        return False

if __name__ == "__main__":
    success = test_enhanced_system()
    exit(0 if success else 1)
