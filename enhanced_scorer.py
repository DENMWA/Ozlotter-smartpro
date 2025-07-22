"""
Enhanced scoring system integrating all advanced prediction methods
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import Counter
from config import *
from utils import setup_logger
from pattern_analyzer import PatternAnalyzer
from chaos_analyzer import ChaosAnalyzer
from weight_optimizer import WeightOptimizer
from market_psychology import MarketPsychologyAnalyzer

logger = setup_logger(__name__)

class EnhancedScorer:
    def __init__(self):
        self.pattern_analyzer = PatternAnalyzer()
        self.chaos_analyzer = ChaosAnalyzer()
        self.weight_optimizer = WeightOptimizer()
        self.psychology_analyzer = MarketPsychologyAnalyzer()
        self.weights = self._load_or_default_weights()
        
    def score_predictions_enhanced(self, predictions: List[List[int]], 
                                 historical_df: pd.DataFrame,
                                 include_neural: bool = False,
                                 neural_predictions: Optional[List[List[int]]] = None) -> pd.DataFrame:
        """Enhanced scoring using all available methods"""
        logger.info("Scoring predictions with enhanced methods...")
        
        scored = []
        
        if not self.pattern_analyzer.temporal_patterns:
            self.pattern_analyzer.analyze_temporal_patterns(historical_df)
        if not self.pattern_analyzer.positional_patterns:
            self.pattern_analyzer.analyze_positional_patterns(historical_df)
        if self.pattern_analyzer.correlation_matrix is None:
            self.pattern_analyzer.build_correlation_matrix(historical_df)
        
        chaos_metrics = self.chaos_analyzer.analyze_chaos_metrics(historical_df)
        
        freq_map = self._get_frequency_map(historical_df)
        
        for idx, prediction in enumerate(predictions):
            entry = {
                "ID": f"Enhanced-{idx+1}",
                "Prediction": prediction,
            }
            
            entry["Entropy"] = self._entropy_score(prediction)
            entry["FreqScore"] = self._frequency_score(prediction, freq_map)
            entry["GapScore"] = self._gap_score(prediction)
            
            entry["TemporalScore"] = self._temporal_score(prediction, historical_df)
            entry["CorrelationScore"] = self._correlation_score(prediction)
            entry["ChaosScore"] = self._chaos_score(prediction, chaos_metrics)
            entry["PsychologyScore"] = self._psychology_score(prediction)
            
            if include_neural and neural_predictions:
                entry["NeuralScore"] = self._neural_alignment_score(prediction, neural_predictions)
            else:
                entry["NeuralScore"] = 0.5  # Neutral score
            
            entry["EnhancedScore"] = self._calculate_enhanced_score(entry)
            
            entry["PayoutMultiplier"] = self.psychology_analyzer.calculate_expected_payout_multiplier(prediction)
            
            scored.append(entry)
        
        df = pd.DataFrame(scored)
        return df.sort_values(by="EnhancedScore", ascending=False)
    
    def _calculate_enhanced_score(self, entry: Dict) -> float:
        """Calculate weighted composite score"""
        score = (
            self.weights['entropy_weight'] * entry["Entropy"] +
            self.weights['frequency_weight'] * (entry["FreqScore"] / 100.0) +
            self.weights['gap_weight'] * entry["GapScore"] +
            self.weights['temporal_weight'] * entry["TemporalScore"] +
            self.weights['correlation_weight'] * entry["CorrelationScore"] +
            self.weights['neural_weight'] * entry["NeuralScore"] +
            self.weights['chaos_weight'] * entry["ChaosScore"]
        )
        
        psychology_bonus = entry["PsychologyScore"] * 0.1
        
        return score + psychology_bonus
    
    def _entropy_score(self, number_set: List[int]) -> float:
        """Calculate normalized entropy of a set"""
        counts = Counter(number_set)
        probs = [count / len(number_set) for count in counts.values()]
        return -sum(p * np.log2(p) for p in probs)
    
    def _frequency_score(self, number_set: List[int], frequency_map: Dict[int, int]) -> float:
        """Score set based on frequency weightings"""
        return sum(frequency_map.get(n, 0) for n in number_set)
    
    def _gap_score(self, number_set: List[int]) -> float:
        """Measure gaps between sorted numbers"""
        sorted_nums = sorted(number_set)
        gaps = [b - a for a, b in zip(sorted_nums[:-1], sorted_nums[1:])]
        ideal_gap = MAX_NUMBER / OZLOTTO_NUMBERS
        return float(-np.std([g - ideal_gap for g in gaps]))
    
    def _temporal_score(self, prediction: List[int], historical_df: pd.DataFrame) -> float:
        """Score based on temporal patterns"""
        if not self.pattern_analyzer.temporal_patterns:
            return 0.5
        
        score = 0.0
        count = 0
        
        if 'intervals' in self.pattern_analyzer.temporal_patterns:
            intervals = self.pattern_analyzer.temporal_patterns['intervals']
            for num in prediction:
                if num in intervals:
                    std_interval = intervals[num].get('std_interval', 10)
                    score += 1.0 / (1.0 + std_interval / 10.0)  # Lower std = higher score
                    count += 1
        
        return score / count if count > 0 else 0.5
    
    def _correlation_score(self, prediction: List[int]) -> float:
        """Score based on number correlations"""
        if self.pattern_analyzer.correlation_matrix is None:
            return 0.5
        
        correlation_sum = 0.0
        pair_count = 0
        
        for i, num1 in enumerate(prediction):
            for j, num2 in enumerate(prediction[i+1:], i+1):
                if 1 <= num1 <= MAX_NUMBER and 1 <= num2 <= MAX_NUMBER:
                    correlation = self.pattern_analyzer.correlation_matrix[num1-1][num2-1]
                    correlation_sum += correlation
                    pair_count += 1
        
        return correlation_sum / pair_count if pair_count > 0 else 0.5
    
    def _chaos_score(self, prediction: List[int], chaos_metrics: Dict) -> float:
        """Score based on chaos theory metrics"""
        if not chaos_metrics:
            return 0.5
        
        chaos_scores = []
        for num in prediction:
            if num in chaos_metrics:
                chaos_scores.append(chaos_metrics[num]['chaos_score'])
        
        return float(np.mean(chaos_scores)) if chaos_scores else 0.5
    
    def _psychology_score(self, prediction: List[int]) -> float:
        """Score based on market psychology (anti-popular bonus)"""
        payout_multiplier = self.psychology_analyzer.calculate_expected_payout_multiplier(prediction)
        
        return min(1.0, (payout_multiplier - 1.0) * 2.0)
    
    def _neural_alignment_score(self, prediction: List[int], neural_predictions: List[List[int]]) -> float:
        """Score based on alignment with neural network predictions"""
        if not neural_predictions:
            return 0.5
        
        max_overlap = 0
        for neural_pred in neural_predictions:
            overlap = len(set(prediction) & set(neural_pred))
            max_overlap = max(max_overlap, overlap)
        
        return max_overlap / OZLOTTO_NUMBERS
    
    def _get_frequency_map(self, historical_df: pd.DataFrame) -> Dict[int, int]:
        """Generate frequency count from historical draw data"""
        flat_numbers = historical_df.values.flatten()
        freq = dict(Counter(flat_numbers))
        return freq
    
    def _load_or_default_weights(self) -> Dict[str, float]:
        """Load optimized weights or use defaults"""
        loaded_weights = self.weight_optimizer.load_weights()
        if loaded_weights:
            return loaded_weights
        return self.weight_optimizer._get_default_weights()
    
    def set_weights(self, weights: Dict[str, float]):
        """Set custom weights for scoring"""
        self.weights = weights
    
    def optimize_weights_for_data(self, historical_data: pd.DataFrame):
        """Optimize weights based on historical data"""
        logger.info("Optimizing weights for current dataset...")
        optimized_weights = self.weight_optimizer.optimize_weights(historical_data)
        self.weights = optimized_weights
        return optimized_weights
