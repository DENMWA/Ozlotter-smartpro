"""
Master prediction engine that orchestrates all prediction methods
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from config import *
from utils import setup_logger
from neural_predictor import NeuralPredictionEngine
from advanced_evolution import AdaptiveEvolutionEngine
from pattern_analyzer import PatternAnalyzer
from chaos_analyzer import ChaosAnalyzer
from cross_lottery_analyzer import CrossLotteryAnalyzer
from market_psychology import MarketPsychologyAnalyzer
from weight_optimizer import WeightOptimizer
from enhanced_scorer import EnhancedScorer

logger = setup_logger(__name__)

class MasterPredictor:
    def __init__(self):
        self.neural_engine = NeuralPredictionEngine()
        self.evolution_engine = AdaptiveEvolutionEngine()
        self.pattern_analyzer = PatternAnalyzer()
        self.chaos_analyzer = ChaosAnalyzer()
        self.cross_lottery_analyzer = CrossLotteryAnalyzer()
        self.psychology_analyzer = MarketPsychologyAnalyzer()
        self.weight_optimizer = WeightOptimizer()
        self.enhanced_scorer = EnhancedScorer()
        
    def generate_ensemble_predictions(self, draws_df: pd.DataFrame, 
                                    seed_sets: List[List[int]],
                                    n_predictions: int = 100,
                                    enable_neural: bool = True,
                                    enable_chaos: bool = True,
                                    enable_patterns: bool = True,
                                    enable_psychology: bool = True) -> Dict:
        """Generate predictions using ensemble of all methods"""
        logger.info(f"Generating {n_predictions} ensemble predictions...")
        
        all_predictions = []
        method_contributions = {
            'genetic_evolution': 0,
            'neural_networks': 0,
            'pattern_analysis': 0,
            'chaos_theory': 0,
            'cross_lottery': 0,
            'psychology': 0
        }
        
        method_weights = self._calculate_method_weights(
            draws_df, enable_neural, enable_chaos, enable_patterns, enable_psychology
        )
        
        predictions_by_method = {}
        
        genetic_count = max(1, int(n_predictions * method_weights['genetic']))
        genetic_predictions = self._generate_genetic_predictions(seed_sets, draws_df, genetic_count)
        predictions_by_method['genetic'] = genetic_predictions
        method_contributions['genetic_evolution'] = len(genetic_predictions)
        
        if enable_neural and method_weights['neural'] > 0:
            neural_count = max(1, int(n_predictions * method_weights['neural']))
            neural_predictions = self._generate_neural_predictions(draws_df, neural_count)
            predictions_by_method['neural'] = neural_predictions
            method_contributions['neural_networks'] = len(neural_predictions)
        
        if enable_patterns and method_weights['patterns'] > 0:
            pattern_count = max(1, int(n_predictions * method_weights['patterns']))
            pattern_predictions = self.pattern_analyzer.predict_based_on_patterns(draws_df, pattern_count)
            predictions_by_method['patterns'] = pattern_predictions
            method_contributions['pattern_analysis'] = len(pattern_predictions)
        
        if enable_chaos and method_weights['chaos'] > 0:
            chaos_count = max(1, int(n_predictions * method_weights['chaos']))
            chaos_predictions = self.chaos_analyzer.predict_using_chaos(draws_df, chaos_count)
            predictions_by_method['chaos'] = chaos_predictions
            method_contributions['chaos_theory'] = len(chaos_predictions)
        
        if method_weights['cross_lottery'] > 0:
            cross_count = max(1, int(n_predictions * method_weights['cross_lottery']))
            base_predictions = genetic_predictions[:cross_count] if genetic_predictions else []
            if base_predictions:
                global_patterns = self.cross_lottery_analyzer.analyze_global_patterns(draws_df)
                cross_predictions = self.cross_lottery_analyzer.apply_global_insights_to_prediction(
                    base_predictions, global_patterns
                )
                predictions_by_method['cross_lottery'] = cross_predictions
                method_contributions['cross_lottery'] = len(cross_predictions)
        
        for method_predictions in predictions_by_method.values():
            all_predictions.extend(method_predictions)
        
        unique_predictions = self._remove_duplicates_preserve_diversity(all_predictions)
        
        if len(unique_predictions) > n_predictions:
            unique_predictions = unique_predictions[:n_predictions]
        
        if enable_psychology:
            unique_predictions = self.psychology_analyzer.apply_anti_popular_strategy(
                unique_predictions, intensity=0.3
            )
            method_contributions['psychology'] = len(unique_predictions)
        
        scored_df = self.enhanced_scorer.score_predictions_enhanced(
            unique_predictions, 
            draws_df,
            include_neural=enable_neural,
            neural_predictions=predictions_by_method.get('neural', [])
        )
        
        return {
            'predictions': unique_predictions,
            'scored_dataframe': scored_df,
            'method_contributions': method_contributions,
            'method_weights': method_weights,
            'ensemble_info': {
                'total_methods_used': len([k for k, v in method_contributions.items() if v > 0]),
                'diversity_score': self._calculate_ensemble_diversity(unique_predictions),
                'average_payout_multiplier': np.mean([
                    self.psychology_analyzer.calculate_expected_payout_multiplier(pred) 
                    for pred in unique_predictions
                ])
            }
        }
    
    def _calculate_method_weights(self, draws_df: pd.DataFrame, 
                                enable_neural: bool, enable_chaos: bool, 
                                enable_patterns: bool, enable_psychology: bool) -> Dict[str, float]:
        """Calculate weights for each prediction method"""
        weights = {
            'genetic': 0.4,  # Always primary method
            'neural': 0.2 if enable_neural else 0.0,
            'patterns': 0.15 if enable_patterns else 0.0,
            'chaos': 0.1 if enable_chaos else 0.0,
            'cross_lottery': 0.1,
            'psychology': 0.05 if enable_psychology else 0.0
        }
        
        data_size = len(draws_df)
        if data_size < 50:
            weights['genetic'] = 0.6
            weights['neural'] *= 0.5
            weights['patterns'] *= 0.7
        elif data_size > 200:
            weights['neural'] *= 1.2
            weights['patterns'] *= 1.1
            weights['chaos'] *= 1.3
        
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _generate_genetic_predictions(self, seed_sets: List[List[int]], 
                                    draws_df: pd.DataFrame, count: int) -> List[List[int]]:
        """Generate predictions using genetic evolution"""
        if not seed_sets:
            seed_sets = []
            for _ in range(5):
                seed = sorted(np.random.choice(range(MIN_NUMBER, MAX_NUMBER + 1), OZLOTTO_NUMBERS, replace=False))
                seed_sets.append(seed)
        
        flat_numbers = draws_df.values.flatten()
        mutation_pool = list(pd.Series(flat_numbers).value_counts().head(20).index)
        
        predictions = self.evolution_engine.island_model_evolution(seed_sets, mutation_pool, count)
        
        return predictions
    
    def _generate_neural_predictions(self, draws_df: pd.DataFrame, count: int) -> List[List[int]]:
        """Generate predictions using neural networks"""
        try:
            if not self.neural_engine.is_trained():
                logger.info("Training neural network...")
                self.neural_engine.train_lstm_model(draws_df)
            
            predictions = self.neural_engine.generate_lstm_predictions(count)
            
            valid_predictions = []
            for pred in predictions:
                if len(set(pred)) == OZLOTTO_NUMBERS and all(MIN_NUMBER <= x <= MAX_NUMBER for x in pred):
                    valid_predictions.append(sorted(pred))
            
            return valid_predictions
            
        except Exception as e:
            logger.warning(f"Neural prediction failed: {e}")
            return []
    
    def _remove_duplicates_preserve_diversity(self, predictions: List[List[int]]) -> List[List[int]]:
        """Remove duplicates while preserving diversity"""
        unique_predictions = []
        seen_signatures = set()
        
        diversity_scored = []
        for pred in predictions:
            signature = tuple(sorted(pred))
            if signature not in seen_signatures:
                diversity_score = self._calculate_prediction_diversity_score(pred, unique_predictions)
                diversity_scored.append((pred, diversity_score, signature))
        
        diversity_scored.sort(key=lambda x: x[1], reverse=True)
        
        for pred, score, signature in diversity_scored:
            if signature not in seen_signatures:
                unique_predictions.append(pred)
                seen_signatures.add(signature)
        
        return unique_predictions
    
    def _calculate_prediction_diversity_score(self, prediction: List[int], 
                                            existing_predictions: List[List[int]]) -> float:
        """Calculate diversity score for a prediction"""
        if not existing_predictions:
            return 1.0
        
        min_similarity = float('inf')
        for existing in existing_predictions:
            similarity = len(set(prediction) & set(existing)) / OZLOTTO_NUMBERS
            min_similarity = min(min_similarity, similarity)
        
        return 1.0 - min_similarity
    
    def _calculate_ensemble_diversity(self, predictions: List[List[int]]) -> float:
        """Calculate overall diversity of the ensemble"""
        if len(predictions) < 2:
            return 1.0
        
        total_similarity = 0.0
        pair_count = 0
        
        for i, pred1 in enumerate(predictions):
            for j, pred2 in enumerate(predictions[i+1:], i+1):
                similarity = len(set(pred1) & set(pred2)) / OZLOTTO_NUMBERS
                total_similarity += similarity
                pair_count += 1
        
        avg_similarity = total_similarity / pair_count if pair_count > 0 else 0
        return 1.0 - avg_similarity
    
    def optimize_ensemble_weights(self, historical_data: pd.DataFrame) -> Dict[str, float]:
        """Optimize weights for ensemble methods"""
        logger.info("Optimizing ensemble weights...")
        
        optimized_weights = self.weight_optimizer.optimize_weights(historical_data)
        self.enhanced_scorer.set_weights(optimized_weights)
        
        return optimized_weights
    
    def get_prediction_insights(self, predictions: List[List[int]], 
                              draws_df: pd.DataFrame) -> Dict:
        """Get comprehensive insights about the predictions"""
        insights = {
            'statistical_analysis': self._analyze_prediction_statistics(predictions),
            'pattern_analysis': self.pattern_analyzer.analyze_temporal_patterns(draws_df),
            'psychology_analysis': self.psychology_analyzer.analyze_prediction_popularity(predictions),
            'chaos_metrics': self.chaos_analyzer.analyze_chaos_metrics(draws_df),
            'diversity_metrics': {
                'ensemble_diversity': self._calculate_ensemble_diversity(predictions),
                'number_distribution': self._analyze_number_distribution(predictions),
                'gap_analysis': self._analyze_gap_distribution(predictions)
            }
        }
        
        return insights
    
    def _analyze_prediction_statistics(self, predictions: List[List[int]]) -> Dict:
        """Analyze statistical properties of predictions"""
        all_numbers = [num for pred in predictions for num in pred]
        
        return {
            'total_predictions': len(predictions),
            'unique_numbers_used': len(set(all_numbers)),
            'most_frequent_numbers': pd.Series(all_numbers).value_counts().head(10).to_dict(),
            'average_sum': np.mean([sum(pred) for pred in predictions]),
            'sum_std': np.std([sum(pred) for pred in predictions]),
            'even_odd_ratio': np.mean([sum(1 for x in pred if x % 2 == 0) / len(pred) for pred in predictions])
        }
    
    def _analyze_number_distribution(self, predictions: List[List[int]]) -> Dict:
        """Analyze distribution of numbers across predictions"""
        all_numbers = [num for pred in predictions for num in pred]
        
        low_range = [x for x in all_numbers if 1 <= x <= 15]
        mid_range = [x for x in all_numbers if 16 <= x <= 31]
        high_range = [x for x in all_numbers if 32 <= x <= 47]
        
        total = len(all_numbers)
        
        return {
            'low_range_percentage': len(low_range) / total * 100 if total > 0 else 0,
            'mid_range_percentage': len(mid_range) / total * 100 if total > 0 else 0,
            'high_range_percentage': len(high_range) / total * 100 if total > 0 else 0,
            'distribution_balance': np.std([len(low_range), len(mid_range), len(high_range)])
        }
    
    def _analyze_gap_distribution(self, predictions: List[List[int]]) -> Dict:
        """Analyze gap distribution in predictions"""
        all_gaps = []
        
        for pred in predictions:
            sorted_pred = sorted(pred)
            gaps = [sorted_pred[i+1] - sorted_pred[i] for i in range(len(sorted_pred)-1)]
            all_gaps.extend(gaps)
        
        if all_gaps:
            return {
                'average_gap': np.mean(all_gaps),
                'gap_std': np.std(all_gaps),
                'min_gap': min(all_gaps),
                'max_gap': max(all_gaps),
                'ideal_gap': MAX_NUMBER / OZLOTTO_NUMBERS,
                'gap_consistency': 1.0 / (1.0 + np.std(all_gaps))
            }
        
        return {'status': 'No gap data available'}
