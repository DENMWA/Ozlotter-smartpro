"""
Dynamic weight optimization system for scoring functions
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy.optimize import minimize, differential_evolution
import json
import os
from config import *
from utils import setup_logger

logger = setup_logger(__name__)

class WeightOptimizer:
    def __init__(self):
        self.optimal_weights = None
        self.optimization_history = []
        self.performance_metrics = {}
        
    def optimize_weights(self, historical_data: pd.DataFrame, 
                        validation_period: int = 50,
                        method: str = 'differential_evolution') -> Dict[str, float]:
        """Optimize scoring weights using historical performance"""
        logger.info(f"Optimizing weights using {method}...")
        
        if len(historical_data) < validation_period + 20:
            logger.warning("Insufficient data for weight optimization")
            return self._get_default_weights()
        
        train_data = historical_data[:-validation_period]
        validation_data = historical_data[-validation_period:]
        
        bounds = [
            (0.0, 1.0),  # entropy_weight
            (0.0, 1.0),  # frequency_weight
            (0.0, 1.0),  # gap_weight
            (0.0, 1.0),  # temporal_weight
            (0.0, 1.0),  # correlation_weight
            (0.0, 1.0),  # neural_weight
            (0.0, 1.0),  # chaos_weight
        ]
        
        def objective(weights):
            return -self._evaluate_weights(weights, train_data, validation_data)
        
        try:
            if method == 'differential_evolution':
                result = differential_evolution(
                    objective, 
                    bounds, 
                    seed=42,
                    maxiter=50,
                    popsize=10
                )
            else:
                initial_weights = np.array([1/7] * 7)
                result = minimize(
                    objective,
                    initial_weights,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 50}
                )
            
            if result.success:
                optimal_weights = result.x
                optimal_weights = optimal_weights / np.sum(optimal_weights)
                
                self.optimal_weights = {
                    'entropy_weight': optimal_weights[0],
                    'frequency_weight': optimal_weights[1],
                    'gap_weight': optimal_weights[2],
                    'temporal_weight': optimal_weights[3],
                    'correlation_weight': optimal_weights[4],
                    'neural_weight': optimal_weights[5],
                    'chaos_weight': optimal_weights[6]
                }
                
                self._save_weights()
                
                logger.info(f"Weight optimization completed. Best score: {-result.fun:.4f}")
                return self.optimal_weights
            else:
                logger.warning("Weight optimization failed, using default weights")
                return self._get_default_weights()
                
        except Exception as e:
            logger.error(f"Error in weight optimization: {e}")
            return self._get_default_weights()
    
    def _evaluate_weights(self, weights: np.ndarray, 
                         train_data: pd.DataFrame, 
                         validation_data: pd.DataFrame) -> float:
        """Evaluate weight configuration using backtesting"""
        try:
            weights = weights / np.sum(weights)
            
            total_score = 0.0
            num_tests = min(5, len(validation_data))
            
            for i in range(num_tests):
                historical_subset = pd.concat([train_data, validation_data.iloc[:i]])
                
                if len(historical_subset) < 10:
                    continue
                
                test_predictions = self._generate_test_predictions(historical_subset)
                
                if i < len(validation_data):
                    actual_draw = validation_data.iloc[i].values.tolist()
                    prediction_score = self._score_prediction_accuracy(test_predictions, actual_draw)
                    total_score += prediction_score
            
            return total_score / num_tests if num_tests > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Error evaluating weights: {e}")
            return 0.0
    
    def _generate_test_predictions(self, historical_data: pd.DataFrame, n_predictions: int = 5) -> List[List[int]]:
        """Generate test predictions for weight evaluation"""
        predictions = []
        
        flat_numbers = historical_data.values.flatten()
        unique, counts = np.unique(flat_numbers, return_counts=True)
        freq_dict = dict(zip(unique, counts))
        
        for _ in range(n_predictions):
            prediction = []
            
            available_numbers = list(range(MIN_NUMBER, MAX_NUMBER + 1))
            
            for _ in range(OZLOTTO_NUMBERS):
                if available_numbers:
                    weights = [freq_dict.get(num, 1) for num in available_numbers]
                    weights = np.array(weights) / np.sum(weights)
                    
                    selected = np.random.choice(available_numbers, p=weights)
                    prediction.append(selected)
                    available_numbers.remove(selected)
            
            predictions.append(sorted(prediction))
        
        return predictions
    
    def _score_prediction_accuracy(self, predictions: List[List[int]], actual_draw: List[int]) -> float:
        """Score prediction accuracy against actual draw"""
        best_score = 0.0
        
        for prediction in predictions:
            matches = len(set(prediction) & set(actual_draw))
            if matches == 7:
                score = 1000.0
            elif matches == 6:
                score = 100.0
            elif matches == 5:
                score = 10.0
            elif matches == 4:
                score = 5.0
            elif matches == 3:
                score = 2.0
            elif matches == 2:
                score = 1.0
            else:
                score = 0.0
            
            best_score = max(best_score, score)
        
        return best_score
    
    def load_weights(self) -> Optional[Dict[str, float]]:
        """Load previously optimized weights"""
        try:
            if os.path.exists(WEIGHTS_FILE):
                with open(WEIGHTS_FILE, 'r') as f:
                    self.optimal_weights = json.load(f)
                logger.info("Loaded optimized weights from file")
                return self.optimal_weights
        except Exception as e:
            logger.warning(f"Error loading weights: {e}")
        
        return None
    
    def _save_weights(self):
        """Save optimized weights to file"""
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(WEIGHTS_FILE, 'w') as f:
                json.dump(self.optimal_weights, f, indent=2)
            logger.info("Saved optimized weights to file")
        except Exception as e:
            logger.error(f"Error saving weights: {e}")
    
    def _get_default_weights(self) -> Dict[str, float]:
        """Get default weight configuration"""
        return {
            'entropy_weight': DEFAULT_ENTROPY_WEIGHT,
            'frequency_weight': DEFAULT_FREQUENCY_WEIGHT,
            'gap_weight': DEFAULT_GAP_WEIGHT,
            'temporal_weight': DEFAULT_TEMPORAL_WEIGHT,
            'correlation_weight': DEFAULT_CORRELATION_WEIGHT,
            'neural_weight': 0.1,
            'chaos_weight': 0.1
        }
