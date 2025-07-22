"""
Cross-lottery intelligence for global pattern mining
"""
import numpy as np
import pandas as pd
import requests
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from config import *
from utils import setup_logger

logger = setup_logger(__name__)

class CrossLotteryAnalyzer:
    def __init__(self):
        self.global_patterns = {}
        self.lottery_configs = {
            'oz_lotto': {'numbers': 7, 'max_num': 47, 'min_num': 1},
            'euro_millions': {'numbers': 5, 'max_num': 50, 'min_num': 1},
            'powerball_main': {'numbers': 5, 'max_num': 69, 'min_num': 1},
            'lotto_max': {'numbers': 7, 'max_num': 50, 'min_num': 1}
        }
        
    def analyze_global_patterns(self, oz_lotto_data: pd.DataFrame) -> Dict:
        """Analyze patterns across multiple lottery systems"""
        logger.info("Analyzing global lottery patterns...")
        
        global_insights = {}
        
        oz_patterns = self._extract_universal_patterns(oz_lotto_data, 'oz_lotto')
        global_insights['oz_lotto'] = oz_patterns
        
        for lottery_name, config in self.lottery_configs.items():
            if lottery_name != 'oz_lotto':
                simulated_data = self._simulate_lottery_data(config, 100)
                patterns = self._extract_universal_patterns(simulated_data, lottery_name)
                global_insights[lottery_name] = patterns
        
        universal_patterns = self._find_universal_patterns(global_insights)
        
        return {
            'individual_patterns': global_insights,
            'universal_patterns': universal_patterns,
            'cross_lottery_insights': self._generate_cross_lottery_insights(global_insights)
        }
    
    def _extract_universal_patterns(self, data: pd.DataFrame, lottery_name: str) -> Dict:
        """Extract patterns that could be universal across lotteries"""
        patterns = {}
        
        config = self.lottery_configs.get(lottery_name, self.lottery_configs['oz_lotto'])
        
        flat_numbers = data.values.flatten()
        
        normalized_numbers = (flat_numbers - config['min_num']) / (config['max_num'] - config['min_num'])
        
        patterns['distribution'] = {
            'low_third': np.mean(normalized_numbers < 0.33),
            'mid_third': np.mean((normalized_numbers >= 0.33) & (normalized_numbers < 0.67)),
            'high_third': np.mean(normalized_numbers >= 0.67)
        }
        
        gap_patterns = []
        for _, draw in data.iterrows():
            sorted_draw = sorted(draw.values)
            gaps = [sorted_draw[i+1] - sorted_draw[i] for i in range(len(sorted_draw)-1)]
            normalized_gaps = [g / (config['max_num'] - config['min_num']) for g in gaps]
            gap_patterns.extend(normalized_gaps)
        
        patterns['gaps'] = {
            'mean_gap': np.mean(gap_patterns),
            'std_gap': np.std(gap_patterns),
            'gap_distribution': np.histogram(gap_patterns, bins=10)[0].tolist()
        }
        
        even_odd_ratios = []
        for _, draw in data.iterrows():
            even_count = sum(1 for x in draw.values if x % 2 == 0)
            even_ratio = even_count / len(draw.values)
            even_odd_ratios.append(even_ratio)
        
        patterns['even_odd'] = {
            'mean_even_ratio': np.mean(even_odd_ratios),
            'std_even_ratio': np.std(even_odd_ratios)
        }
        
        draw_sums = [sum(draw.values) for _, draw in data.iterrows()]
        expected_sum = config['numbers'] * (config['max_num'] + config['min_num']) / 2
        normalized_sums = [s / expected_sum for s in draw_sums]
        
        patterns['sums'] = {
            'mean_normalized_sum': np.mean(normalized_sums),
            'std_normalized_sum': np.std(normalized_sums)
        }
        
        return patterns
    
    def _simulate_lottery_data(self, config: Dict, n_draws: int) -> pd.DataFrame:
        """Simulate lottery data for other systems"""
        draws = []
        
        for _ in range(n_draws):
            draw = sorted(np.random.choice(
                range(config['min_num'], config['max_num'] + 1),
                size=config['numbers'],
                replace=False
            ))
            draws.append(draw)
        
        columns = [f"N{i+1}" for i in range(config['numbers'])]
        return pd.DataFrame(draws, columns=columns)
    
    def _find_universal_patterns(self, global_insights: Dict) -> Dict:
        """Find patterns that are consistent across lottery systems"""
        universal = {}
        
        distributions = [insights['distribution'] for insights in global_insights.values()]
        
        universal['distribution_consistency'] = {
            'low_third_variance': np.var([d['low_third'] for d in distributions]),
            'mid_third_variance': np.var([d['mid_third'] for d in distributions]),
            'high_third_variance': np.var([d['high_third'] for d in distributions])
        }
        
        gap_means = [insights['gaps']['mean_gap'] for insights in global_insights.values()]
        universal['gap_consistency'] = {
            'mean_gap_variance': np.var(gap_means),
            'universal_gap_trend': np.mean(gap_means)
        }
        
        even_ratios = [insights['even_odd']['mean_even_ratio'] for insights in global_insights.values()]
        universal['even_odd_consistency'] = {
            'ratio_variance': np.var(even_ratios),
            'universal_even_ratio': np.mean(even_ratios)
        }
        
        return universal
    
    def _generate_cross_lottery_insights(self, global_insights: Dict) -> List[str]:
        """Generate insights from cross-lottery analysis"""
        insights = []
        
        oz_dist = global_insights['oz_lotto']['distribution']
        insights.append(f"Oz Lotto low-third frequency: {oz_dist['low_third']:.3f}")
        insights.append(f"Oz Lotto mid-third frequency: {oz_dist['mid_third']:.3f}")
        insights.append(f"Oz Lotto high-third frequency: {oz_dist['high_third']:.3f}")
        
        oz_gaps = global_insights['oz_lotto']['gaps']
        insights.append(f"Average normalized gap: {oz_gaps['mean_gap']:.3f}")
        
        oz_even_odd = global_insights['oz_lotto']['even_odd']
        insights.append(f"Average even number ratio: {oz_even_odd['mean_even_ratio']:.3f}")
        
        return insights
    
    def apply_global_insights_to_prediction(self, base_predictions: List[List[int]], 
                                          global_patterns: Dict) -> List[List[int]]:
        """Apply global insights to improve predictions"""
        logger.info("Applying global insights to predictions...")
        
        enhanced_predictions = []
        
        for prediction in base_predictions:
            enhanced = self._enhance_prediction_with_global_patterns(prediction, global_patterns)
            enhanced_predictions.append(enhanced)
        
        return enhanced_predictions
    
    def _enhance_prediction_with_global_patterns(self, prediction: List[int], 
                                               global_patterns: Dict) -> List[int]:
        """Enhance a single prediction using global patterns"""
        enhanced = prediction[:]
        
        if 'universal_patterns' not in global_patterns:
            return enhanced
        
        universal = global_patterns['universal_patterns']
        
        if 'even_odd_consistency' in universal:
            target_even_ratio = universal['even_odd_consistency']['universal_even_ratio']
            current_even_count = sum(1 for x in enhanced if x % 2 == 0)
            current_even_ratio = current_even_count / len(enhanced)
            
            if abs(current_even_ratio - target_even_ratio) > 0.2:
                enhanced = self._adjust_even_odd_ratio(enhanced, target_even_ratio)
        
        if 'gap_consistency' in universal:
            target_gap = universal['gap_consistency']['universal_gap_trend']
            enhanced = self._adjust_gaps(enhanced, target_gap)
        
        return enhanced
    
    def _adjust_even_odd_ratio(self, prediction: List[int], target_ratio: float) -> List[int]:
        """Adjust prediction to match target even/odd ratio"""
        current_even_count = sum(1 for x in prediction if x % 2 == 0)
        target_even_count = int(target_ratio * len(prediction))
        
        if current_even_count == target_even_count:
            return prediction
        
        adjusted = prediction[:]
        
        if current_even_count < target_even_count:
            for i, num in enumerate(adjusted):
                if num % 2 == 1:  # Odd number
                    for even_candidate in range(2, MAX_NUMBER + 1, 2):
                        if even_candidate not in adjusted:
                            adjusted[i] = even_candidate
                            current_even_count += 1
                            break
                    if current_even_count >= target_even_count:
                        break
        else:
            for i, num in enumerate(adjusted):
                if num % 2 == 0:  # Even number
                    for odd_candidate in range(1, MAX_NUMBER + 1, 2):
                        if odd_candidate not in adjusted:
                            adjusted[i] = odd_candidate
                            current_even_count -= 1
                            break
                    if current_even_count <= target_even_count:
                        break
        
        return sorted(adjusted)
    
    def _adjust_gaps(self, prediction: List[int], target_gap: float) -> List[int]:
        """Adjust prediction gaps to match target pattern"""
        sorted_pred = sorted(prediction)
        current_gaps = [sorted_pred[i+1] - sorted_pred[i] for i in range(len(sorted_pred)-1)]
        current_mean_gap = np.mean(current_gaps)
        
        if abs(current_mean_gap - target_gap * MAX_NUMBER) < 2:
            return prediction
        
        if current_mean_gap < target_gap * MAX_NUMBER:
            adjusted = []
            for i, num in enumerate(sorted_pred):
                if i == 0:
                    adjusted.append(max(MIN_NUMBER, num - 1))
                else:
                    adjusted.append(min(MAX_NUMBER, num + 1))
        else:
            adjusted = []
            center = np.mean(sorted_pred)
            for num in sorted_pred:
                new_num = int(center + 0.8 * (num - center))
                adjusted.append(max(MIN_NUMBER, min(MAX_NUMBER, new_num)))
        
        unique_adjusted = []
        for num in adjusted:
            if num not in unique_adjusted:
                unique_adjusted.append(num)
        
        while len(unique_adjusted) < OZLOTTO_NUMBERS:
            candidate = np.random.randint(MIN_NUMBER, MAX_NUMBER + 1)
            if candidate not in unique_adjusted:
                unique_adjusted.append(candidate)
        
        return sorted(unique_adjusted[:OZLOTTO_NUMBERS])
