"""
Adaptive seed manager with performance tracking and intelligent seed evolution
"""
import json
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from config import *
from utils import setup_logger

logger = setup_logger(__name__)

class AdaptiveSeedManager:
    def __init__(self):
        self.seed_file = SEED_FILE
        self.performance_file = f"{DATA_DIR}/seed_performance.json"
        self.seed_performance = {}
        self.generation_history = []
        
    def load_seeds_with_performance(self) -> Tuple[List[List[int]], Dict]:
        """Load seeds along with their performance metrics"""
        seeds = self._load_basic_seeds()
        performance = self._load_performance_data()
        
        if performance and seeds:
            seeds = self._filter_seeds_by_performance(seeds, performance)
        
        logger.info(f"Loaded {len(seeds)} seeds with performance data")
        return seeds, performance
    
    def save_seeds_with_performance(self, seed_sets: List[List[int]], 
                                  performance_scores: Optional[List[float]] = None,
                                  generation_info: Optional[Dict] = None) -> List[List[int]]:
        """Save seeds along with performance tracking"""
        existing_seeds, existing_performance = self.load_seeds_with_performance()
        
        all_seeds = existing_seeds + seed_sets
        
        if performance_scores:
            for i, seed in enumerate(seed_sets):
                seed_key = self._seed_to_key(seed)
                if i < len(performance_scores):
                    self._update_seed_performance(seed_key, performance_scores[i], generation_info)
        
        elite_seeds = self._select_elite_seeds(all_seeds, target_count=20)
        
        self._save_basic_seeds(elite_seeds)
        self._save_performance_data()
        
        if generation_info:
            self.generation_history.append({
                'timestamp': pd.Timestamp.now().isoformat(),
                'new_seeds_count': len(seed_sets),
                'elite_seeds_count': len(elite_seeds),
                'generation_info': generation_info
            })
            self._save_generation_history()
        
        logger.info(f"Saved {len(elite_seeds)} elite seeds with performance tracking")
        return elite_seeds
    
    def get_adaptive_mutation_pool(self, historical_draws: pd.DataFrame, 
                                 pool_size: int = DEFAULT_MUTATION_POOL_SIZE) -> List[int]:
        """Generate adaptive mutation pool based on recent performance and trends"""
        flat_numbers = historical_draws.values.flatten()
        frequency_counts = pd.Series(flat_numbers).value_counts()
        
        recent_draws = historical_draws.tail(20) if len(historical_draws) > 20 else historical_draws
        recent_flat = recent_draws.values.flatten()
        recent_counts = pd.Series(recent_flat).value_counts()
        
        trend_scores = {}
        for num in range(MIN_NUMBER, MAX_NUMBER + 1):
            recent_freq = recent_counts.get(num, 0) / len(recent_flat) if len(recent_flat) > 0 else 0
            overall_freq = frequency_counts.get(num, 0) / len(flat_numbers) if len(flat_numbers) > 0 else 0
            
            trend_scores[num] = recent_freq - overall_freq
        
        adaptive_scores = {}
        for num in range(MIN_NUMBER, MAX_NUMBER + 1):
            freq_score = frequency_counts.get(num, 0)
            trend_score = trend_scores[num]
            
            adaptive_scores[num] = 0.7 * freq_score + 0.3 * trend_score * 1000
        
        sorted_numbers = sorted(adaptive_scores.items(), key=lambda x: x[1], reverse=True)
        mutation_pool = [num for num, score in sorted_numbers[:pool_size]]
        
        logger.info(f"Generated adaptive mutation pool with {len(mutation_pool)} numbers")
        return mutation_pool
    
    def _load_basic_seeds(self) -> List[List[int]]:
        """Load basic seeds from JSON file"""
        if not os.path.exists(self.seed_file):
            return []
        
        try:
            with open(self.seed_file, "r") as f:
                seeds = json.load(f)
            return [s for s in seeds if self._validate_seed(s)]
        except Exception as e:
            logger.error(f"Error loading seeds: {e}")
            return []
    
    def _load_performance_data(self) -> Dict:
        """Load performance tracking data"""
        if not os.path.exists(self.performance_file):
            return {}
        
        try:
            with open(self.performance_file, "r") as f:
                self.seed_performance = json.load(f)
            return self.seed_performance
        except Exception as e:
            logger.error(f"Error loading performance data: {e}")
            return {}
    
    def _save_basic_seeds(self, seeds: List[List[int]]):
        """Save basic seeds to JSON file"""
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(self.seed_file, "w") as f:
                json.dump(seeds, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving seeds: {e}")
    
    def _save_performance_data(self):
        """Save performance tracking data"""
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(self.performance_file, "w") as f:
                json.dump(self.seed_performance, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving performance data: {e}")
    
    def _save_generation_history(self):
        """Save generation history"""
        try:
            history_file = f"{DATA_DIR}/generation_history.json"
            with open(history_file, "w") as f:
                json.dump(self.generation_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving generation history: {e}")
    
    def _seed_to_key(self, seed: List[int]) -> str:
        """Convert seed to string key for tracking"""
        return ",".join(map(str, sorted(seed)))
    
    def _update_seed_performance(self, seed_key: str, score: float, generation_info: Optional[Dict]):
        """Update performance tracking for a seed"""
        if seed_key not in self.seed_performance:
            self.seed_performance[seed_key] = {
                'scores': [],
                'generations': [],
                'first_seen': pd.Timestamp.now().isoformat()
            }
        
        self.seed_performance[seed_key]['scores'].append(score)
        if generation_info:
            self.seed_performance[seed_key]['generations'].append(generation_info)
    
    def _filter_seeds_by_performance(self, seeds: List[List[int]], performance: Dict) -> List[List[int]]:
        """Filter seeds based on performance metrics"""
        if not performance:
            return seeds
        
        scored_seeds = []
        for seed in seeds:
            seed_key = self._seed_to_key(seed)
            if seed_key in performance and performance[seed_key]['scores']:
                avg_score = np.mean(performance[seed_key]['scores'])
                scored_seeds.append((seed, avg_score))
            else:
                scored_seeds.append((seed, 0.0))
        
        scored_seeds.sort(key=lambda x: x[1], reverse=True)
        return [seed for seed, score in scored_seeds]
    
    def _select_elite_seeds(self, all_seeds: List[List[int]], target_count: int) -> List[List[int]]:
        """Select elite seeds using adaptive criteria"""
        if len(all_seeds) <= target_count:
            return all_seeds
        
        unique_seeds = []
        seen = set()
        for seed in all_seeds:
            seed_tuple = tuple(sorted(seed))
            if seed_tuple not in seen:
                unique_seeds.append(seed)
                seen.add(seed_tuple)
        
        if len(unique_seeds) <= target_count:
            return unique_seeds
        
        scored_seeds = []
        for seed in unique_seeds:
            seed_key = self._seed_to_key(seed)
            
            if seed_key in self.seed_performance and self.seed_performance[seed_key]['scores']:
                perf_score = np.mean(self.seed_performance[seed_key]['scores'])
            else:
                perf_score = 0.5
            
            diversity_score = self._calculate_diversity_score(seed, unique_seeds)
            
            combined_score = 0.7 * perf_score + 0.3 * diversity_score
            scored_seeds.append((seed, combined_score))
        
        scored_seeds.sort(key=lambda x: x[1], reverse=True)
        return [seed for seed, score in scored_seeds[:target_count]]
    
    def _calculate_diversity_score(self, seed: List[int], all_seeds: List[List[int]]) -> float:
        """Calculate diversity score for a seed"""
        if len(all_seeds) <= 1:
            return 1.0
        
        similarities = []
        for other_seed in all_seeds:
            if other_seed != seed:
                similarity = len(set(seed) & set(other_seed)) / OZLOTTO_NUMBERS
                similarities.append(similarity)
        
        avg_similarity = np.mean(similarities) if similarities else 0
        return 1.0 - avg_similarity
    
    def _validate_seed(self, seed: List[int]) -> bool:
        """Validate seed format"""
        return (len(seed) == OZLOTTO_NUMBERS and 
                len(set(seed)) == OZLOTTO_NUMBERS and
                all(MIN_NUMBER <= x <= MAX_NUMBER for x in seed))
