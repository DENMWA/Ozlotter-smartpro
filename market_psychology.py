"""
Market psychology and anti-popular number strategies
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import Counter
from config import *
from utils import setup_logger

logger = setup_logger(__name__)

class MarketPsychologyAnalyzer:
    def __init__(self):
        self.popular_patterns = {
            'sequential': [[1,2,3,4,5,6,7], [2,3,4,5,6,7,8], [40,41,42,43,44,45,46]],
            'birthdate_heavy': [[1,2,3,4,5,6,7], [10,11,12,13,14,15,16], [20,21,22,23,24,25,26]],
            'multiples_of_7': [[7,14,21,28,35,42,47]],
            'lucky_numbers': [[3,7,13,21,27,33,39], [1,8,15,22,29,36,43]],
            'geometric': [[1,4,9,16,25,36,47], [2,8,18,32,47,46,45]]
        }
        self.popularity_weights = {}
        
    def analyze_popular_patterns(self, historical_data: pd.DataFrame) -> Dict:
        """Analyze which patterns appear to be popular based on frequency"""
        logger.info("Analyzing popular number patterns...")
        
        pattern_analysis = {}
        
        for pattern_name, patterns in self.popular_patterns.items():
            pattern_frequency = 0
            total_possible = len(historical_data)
            
            for pattern in patterns:
                for _, draw in historical_data.iterrows():
                    draw_set = set(draw.values)
                    pattern_set = set(pattern)
                    
                    overlap = len(draw_set.intersection(pattern_set))
                    if overlap >= 4:
                        pattern_frequency += 1
            
            pattern_analysis[pattern_name] = {
                'frequency': pattern_frequency,
                'relative_frequency': pattern_frequency / total_possible if total_possible > 0 else 0,
                'popularity_score': self._calculate_popularity_score(pattern_name, pattern_frequency)
            }
        
        return pattern_analysis
    
    def calculate_number_popularity(self, number_usage_data: Optional[Dict] = None) -> Dict[int, float]:
        """Calculate popularity scores for individual numbers"""
        logger.info("Calculating number popularity scores...")
        
        popularity_scores = {}
        
        for num in range(MIN_NUMBER, MAX_NUMBER + 1):
            score = 0.0
            
            if 1 <= num <= 31:
                score += 0.3
            
            if num in [3, 7, 13, 21, 27, 33, 39]:
                score += 0.2
            
            if num % 5 == 0 or num % 10 == 0:
                score += 0.1
            
            if num > 40:
                score -= 0.1
            
            if num % 10 == 0:
                score += 0.15
            
            if num == 13:
                score -= 0.1
            
            popularity_scores[num] = max(0.0, min(1.0, score))
        
        return popularity_scores
    
    def apply_anti_popular_strategy(self, predictions: List[List[int]], 
                                   intensity: float = 0.5) -> List[List[int]]:
        """Apply anti-popular strategy to maximize expected payout"""
        logger.info(f"Applying anti-popular strategy with intensity {intensity}...")
        
        popularity_scores = self.calculate_number_popularity()
        
        enhanced_predictions = []
        
        for prediction in predictions:
            if np.random.random() < intensity:
                enhanced = self._make_prediction_less_popular(prediction, popularity_scores)
                enhanced_predictions.append(enhanced)
            else:
                enhanced_predictions.append(prediction)
        
        return enhanced_predictions
    
    def calculate_expected_payout_multiplier(self, prediction: List[int]) -> float:
        """Calculate expected payout multiplier based on popularity"""
        popularity_scores = self.calculate_number_popularity()
        
        avg_popularity = np.mean([popularity_scores[num] for num in prediction])
        
        payout_multiplier = 1.0 + (1.0 - avg_popularity) * 0.5
        
        return payout_multiplier
    
    def _calculate_popularity_score(self, pattern_name: str, frequency: int) -> float:
        """Calculate popularity score for a pattern"""
        base_scores = {
            'sequential': 0.8,
            'birthdate_heavy': 0.7,
            'multiples_of_7': 0.4,
            'lucky_numbers': 0.6,
            'geometric': 0.3
        }
        
        base_score = base_scores.get(pattern_name, 0.5)
        
        frequency_adjustment = min(0.3, frequency * 0.01)
        
        return min(1.0, base_score + frequency_adjustment)
    
    def _make_prediction_less_popular(self, prediction: List[int], 
                                    popularity_scores: Dict[int, float]) -> List[int]:
        """Modify prediction to be less popular"""
        enhanced = prediction[:]
        
        prediction_popularity = [(num, popularity_scores[num]) for num in enhanced]
        prediction_popularity.sort(key=lambda x: x[1], reverse=True)
        
        replacements_made = 0
        max_replacements = min(3, len(enhanced) // 2)
        
        for num, popularity in prediction_popularity:
            if replacements_made >= max_replacements:
                break
                
            if popularity > 0.6:
                replacement = self._find_unpopular_replacement(num, enhanced, popularity_scores)
                if replacement:
                    enhanced[enhanced.index(num)] = replacement
                    replacements_made += 1
        
        return sorted(enhanced)
    
    def _find_unpopular_replacement(self, original_num: int, current_prediction: List[int], 
                                  popularity_scores: Dict[int, float]) -> Optional[int]:
        """Find an unpopular replacement for a number"""
        unpopular_candidates = [
            num for num in range(MIN_NUMBER, MAX_NUMBER + 1)
            if num not in current_prediction and popularity_scores[num] < 0.4
        ]
        
        if not unpopular_candidates:
            unpopular_candidates = [
                num for num in range(MIN_NUMBER, MAX_NUMBER + 1)
                if num not in current_prediction
            ]
        
        if unpopular_candidates:
            distances = [(abs(num - original_num), num) for num in unpopular_candidates]
            distances.sort()
            return distances[0][1]
        
        return None
    
    def generate_anti_popular_predictions(self, base_predictions: List[List[int]], 
                                        intensity: float = 0.7) -> List[List[int]]:
        """Generate predictions specifically designed to be anti-popular"""
        logger.info("Generating anti-popular predictions...")
        
        popularity_scores = self.calculate_number_popularity()
        
        unpopular_numbers = sorted(
            range(MIN_NUMBER, MAX_NUMBER + 1),
            key=lambda x: popularity_scores[x]
        )
        
        anti_popular_predictions = []
        
        for base_pred in base_predictions:
            if np.random.random() < intensity:
                anti_popular = []
                
                for num in unpopular_numbers:
                    if len(anti_popular) >= OZLOTTO_NUMBERS:
                        break
                    if num not in anti_popular:
                        anti_popular.append(num)
                
                while len(anti_popular) < OZLOTTO_NUMBERS:
                    candidate = np.random.choice(unpopular_numbers[:20])
                    if candidate not in anti_popular:
                        anti_popular.append(candidate)
                
                anti_popular_predictions.append(sorted(anti_popular[:OZLOTTO_NUMBERS]))
            else:
                anti_popular_predictions.append(base_pred)
        
        return anti_popular_predictions
    
    def analyze_prediction_popularity(self, predictions: List[List[int]]) -> Dict:
        """Analyze the popularity characteristics of predictions"""
        popularity_scores = self.calculate_number_popularity()
        
        analysis = {
            'individual_scores': [],
            'average_popularity': 0.0,
            'popularity_distribution': {'low': 0, 'medium': 0, 'high': 0},
            'expected_payout_multipliers': []
        }
        
        for prediction in predictions:
            pred_popularity = np.mean([popularity_scores[num] for num in prediction])
            payout_multiplier = self.calculate_expected_payout_multiplier(prediction)
            
            analysis['individual_scores'].append(pred_popularity)
            analysis['expected_payout_multipliers'].append(payout_multiplier)
            
            if pred_popularity < 0.3:
                analysis['popularity_distribution']['low'] += 1
            elif pred_popularity < 0.6:
                analysis['popularity_distribution']['medium'] += 1
            else:
                analysis['popularity_distribution']['high'] += 1
        
        analysis['average_popularity'] = np.mean(analysis['individual_scores'])
        
        return analysis
