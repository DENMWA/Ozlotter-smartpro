"""
Chaos theory and fractal analysis for lottery prediction
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from scipy import stats
from config import *
from utils import setup_logger

logger = setup_logger(__name__)

class ChaosAnalyzer:
    def __init__(self):
        self.lyapunov_exponents = {}
        self.fractal_dimensions = {}
        self.entropy_measures = {}
        
    def calculate_lyapunov_exponent(self, sequence: List[int], embedding_dim: int = 3) -> float:
        """Calculate Lyapunov exponent for a number sequence"""
        try:
            ts = np.array(sequence, dtype=float)
            
            if len(ts) < embedding_dim + 1:
                return 0.0
            
            embedded = self._embed_time_series(ts, embedding_dim)
            
            if len(embedded) < 2:
                return 0.0
            
            divergences = []
            
            for i in range(len(embedded) - 1):
                distances = [np.linalg.norm(embedded[i] - embedded[j]) 
                           for j in range(len(embedded)) if j != i]
                
                if not distances:
                    continue
                    
                min_dist = min(distances)
                nearest_idx = distances.index(min_dist) + (1 if distances.index(min_dist) >= i else 0)
                
                if nearest_idx < len(embedded) - 1 and i < len(embedded) - 1:
                    future_dist = np.linalg.norm(embedded[i+1] - embedded[nearest_idx+1])
                    
                    if min_dist > 0 and future_dist > 0:
                        divergence = np.log(future_dist / min_dist)
                        divergences.append(divergence)
            
            return np.mean(divergences) if divergences else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating Lyapunov exponent: {e}")
            return 0.0
    
    def calculate_fractal_dimension(self, sequence: List[int]) -> float:
        """Calculate fractal dimension using box-counting method"""
        try:
            if len(sequence) < 2:
                return 1.0
            
            x = sequence[:-1]
            y = sequence[1:]
            
            scales = np.logspace(0, 2, 20)
            counts = []
            
            for scale in scales:
                x_norm = np.array(x) / scale
                y_norm = np.array(y) / scale
                
                boxes = set()
                for i in range(len(x_norm)):
                    box_x = int(x_norm[i])
                    box_y = int(y_norm[i])
                    boxes.add((box_x, box_y))
                
                counts.append(len(boxes))
            
            log_scales = np.log(scales)
            log_counts = np.log(counts)
            
            valid_idx = np.isfinite(log_scales) & np.isfinite(log_counts) & (np.array(counts) > 0)
            
            if np.sum(valid_idx) < 2:
                return 1.0
            
            slope, _, r_value, _, _ = stats.linregress(log_scales[valid_idx], log_counts[valid_idx])
            
            fractal_dim = -slope
            
            return max(1.0, min(2.0, fractal_dim))
            
        except Exception as e:
            logger.warning(f"Error calculating fractal dimension: {e}")
            return 1.0
    
    def analyze_chaos_metrics(self, draws_df: pd.DataFrame) -> Dict:
        """Analyze chaos metrics for all numbers"""
        logger.info("Analyzing chaos metrics...")
        
        chaos_metrics = {}
        
        for num in range(MIN_NUMBER, MAX_NUMBER + 1):
            sequence = []
            for _, draw in draws_df.iterrows():
                if num in draw.values:
                    sequence.append(1)
                else:
                    sequence.append(0)
            
            if len(sequence) > 10:
                lyapunov = self.calculate_lyapunov_exponent(sequence)
                fractal_dim = self.calculate_fractal_dimension(sequence)
                
                chaos_metrics[num] = {
                    'lyapunov_exponent': lyapunov,
                    'fractal_dimension': fractal_dim,
                    'chaos_score': self._calculate_chaos_score(lyapunov, fractal_dim)
                }
        
        return chaos_metrics
    
    def predict_using_chaos(self, draws_df: pd.DataFrame, n_predictions: int = 5) -> List[List[int]]:
        """Generate predictions using chaos theory principles"""
        logger.info("Generating chaos-based predictions...")
        
        predictions = []
        
        recent_draws = draws_df.tail(50) if len(draws_df) > 50 else draws_df
        chaos_metrics = self.analyze_chaos_metrics(recent_draws)
        
        for _ in range(n_predictions):
            prediction = self._generate_chaos_prediction(chaos_metrics, draws_df)
            if prediction and len(set(prediction)) == OZLOTTO_NUMBERS:
                predictions.append(sorted(prediction))
        
        return predictions
    
    def _embed_time_series(self, ts: np.ndarray, embedding_dim: int, delay: int = 1) -> np.ndarray:
        """Embed time series in higher dimensional space"""
        N = len(ts)
        if N < embedding_dim:
            return np.array([])
        
        embedded = np.zeros((N - (embedding_dim - 1) * delay, embedding_dim))
        
        for i in range(embedding_dim):
            embedded[:, i] = ts[i * delay:N - (embedding_dim - 1 - i) * delay]
        
        return embedded
    
    def _calculate_chaos_score(self, lyapunov: float, fractal_dim: float) -> float:
        """Calculate composite chaos score"""
        lyap_norm = np.tanh(abs(lyapunov))
        fractal_norm = (fractal_dim - 1.0) / 1.0
        
        chaos_score = 0.6 * lyap_norm + 0.4 * fractal_norm
        
        return max(0.0, min(1.0, chaos_score))
    
    def _generate_chaos_prediction(self, chaos_metrics: Dict, draws_df: pd.DataFrame) -> List[int]:
        """Generate prediction based on chaos metrics"""
        prediction = []
        
        chaos_sorted = sorted(chaos_metrics.items(), key=lambda x: x[1]['chaos_score'], reverse=True)
        
        high_chaos_numbers = [num for num, metrics in chaos_sorted[:20]]
        
        for _ in range(OZLOTTO_NUMBERS):
            if high_chaos_numbers:
                weights = [chaos_metrics[num]['fractal_dimension'] for num in high_chaos_numbers]
                weights = np.array(weights) / np.sum(weights)
                
                selected = np.random.choice(high_chaos_numbers, p=weights)
                if selected not in prediction:
                    prediction.append(selected)
                    high_chaos_numbers.remove(selected)
            else:
                num = np.random.randint(MIN_NUMBER, MAX_NUMBER + 1)
                if num not in prediction:
                    prediction.append(num)
        
        return prediction
