"""
Comprehensive backtesting and validation framework
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from config import *
from utils import setup_logger

logger = setup_logger(__name__)

class BacktestingEngine:
    def __init__(self):
        self.results_history = []
        
    def comprehensive_backtest(self, prediction_system, historical_data: pd.DataFrame, 
                             periods: int = 100) -> Dict:
        """Perform comprehensive backtesting of prediction system"""
        logger.info(f"Starting comprehensive backtest over {periods} periods...")
        
        if len(historical_data) < periods + 50:
            logger.warning("Insufficient data for comprehensive backtesting")
            return {"status": "insufficient_data"}
        
        results = []
        
        for i in range(len(historical_data) - periods):
            training_data = historical_data.iloc[:i+50]  # Minimum 50 draws for training
            test_draw = historical_data.iloc[i+50]
            
            if len(training_data) < 20:
                continue
            
            try:
                predictions = self._generate_test_predictions(prediction_system, training_data)
                
                actual_numbers = [int(x) for x in test_draw.values.tolist()]
                performance = self._evaluate_predictions(predictions, actual_numbers)
                
                results.append({
                    'period': i,
                    'training_size': len(training_data),
                    'predictions_count': len(predictions),
                    'best_match': performance['best_match'],
                    'average_match': performance['average_match'],
                    'hit_rate': performance['hit_rate']
                })
                
            except Exception as e:
                logger.warning(f"Backtest period {i} failed: {e}")
                continue
        
        if not results:
            return {"status": "no_valid_results"}
        
        summary = self._calculate_backtest_summary(results)
        
        return {
            "status": "success",
            "periods_tested": len(results),
            "summary": summary,
            "detailed_results": results[-20:],  # Last 20 results
            "statistical_significance": self._test_statistical_significance(results)
        }
    
    def _generate_test_predictions(self, prediction_system, training_data: pd.DataFrame) -> List[List[int]]:
        """Generate predictions for backtesting"""
        flat_numbers = training_data.values.flatten()
        frequency_counts = pd.Series(flat_numbers).value_counts()
        
        predictions = []
        for _ in range(5):  # Generate 5 predictions
            prediction = []
            available_numbers = list(range(MIN_NUMBER, MAX_NUMBER + 1))
            
            weights = [frequency_counts.get(num, 1) for num in available_numbers]
            weights = np.array(weights) / np.sum(weights)
            
            for _ in range(OZLOTTO_NUMBERS):
                if available_numbers:
                    selected = np.random.choice(available_numbers, p=weights)
                    prediction.append(selected)
                    
                    idx = available_numbers.index(selected)
                    available_numbers.pop(idx)
                    weights = np.delete(weights, idx)
                    if len(weights) > 0:
                        weights = weights / np.sum(weights)
            
            predictions.append(sorted(prediction))
        
        return predictions
    
    def _evaluate_predictions(self, predictions: List[List[int]], actual_numbers: List[int]) -> Dict:
        """Evaluate predictions against actual draw"""
        matches = []
        
        for prediction in predictions:
            match_count = len(set(prediction) & set(actual_numbers))
            matches.append(match_count)
        
        return {
            'best_match': max(matches) if matches else 0,
            'average_match': np.mean(matches) if matches else 0,
            'hit_rate': sum(1 for m in matches if m >= 3) / len(matches) if matches else 0
        }
    
    def _calculate_backtest_summary(self, results: List[Dict]) -> Dict:
        """Calculate summary statistics from backtest results"""
        best_matches = [r['best_match'] for r in results]
        average_matches = [r['average_match'] for r in results]
        hit_rates = [r['hit_rate'] for r in results]
        
        return {
            'total_periods': len(results),
            'average_best_match': np.mean(best_matches),
            'std_best_match': np.std(best_matches),
            'max_best_match': max(best_matches),
            'average_hit_rate': np.mean(hit_rates),
            'periods_with_hits': sum(1 for r in hit_rates if r > 0),
            'hit_percentage': sum(1 for r in hit_rates if r > 0) / len(results) * 100
        }
    
    def _test_statistical_significance(self, results: List[Dict]) -> Dict:
        """Test statistical significance of results"""
        best_matches = [r['best_match'] for r in results]
        
        expected_random = OZLOTTO_NUMBERS * (OZLOTTO_NUMBERS / MAX_NUMBER)
        
        from scipy import stats
        
        try:
            t_stat, p_value = stats.ttest_1samp(best_matches, expected_random)
            
            return {
                'expected_random_performance': expected_random,
                'actual_average_performance': np.mean(best_matches),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significantly_better_than_random': p_value < 0.05 and np.mean(best_matches) > expected_random
            }
        except Exception as e:
            logger.warning(f"Statistical significance test failed: {e}")
            return {
                'status': 'test_failed',
                'expected_random_performance': expected_random,
                'actual_average_performance': np.mean(best_matches)
            }
