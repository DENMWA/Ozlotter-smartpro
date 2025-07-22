"""
Multi-dimensional pattern analysis including temporal, positional, and correlation analysis
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import networkx as nx
from config import *
from utils import setup_logger

logger = setup_logger(__name__)

class PatternAnalyzer:
    def __init__(self):
        self.temporal_patterns = {}
        self.positional_patterns = {}
        self.correlation_matrix = None
        self.seasonal_trends = {}
        
    def analyze_temporal_patterns(self, draws_df: pd.DataFrame) -> Dict:
        """Analyze temporal patterns in lottery draws"""
        logger.info("Analyzing temporal patterns...")
        
        if 'date' in draws_df.columns:
            draws_df['date'] = pd.to_datetime(draws_df['date'])
            draws_df['day_of_week'] = draws_df['date'].dt.dayofweek
            draws_df['month'] = draws_df['date'].dt.month
            draws_df['quarter'] = draws_df['date'].dt.quarter
            
            dow_patterns = {}
            for dow in range(7):
                dow_draws = draws_df[draws_df['day_of_week'] == dow]
                if len(dow_draws) > 0:
                    dow_patterns[dow] = self._calculate_frequency_distribution(dow_draws)
            
            monthly_patterns = {}
            for month in range(1, 13):
                month_draws = draws_df[draws_df['month'] == month]
                if len(month_draws) > 0:
                    monthly_patterns[month] = self._calculate_frequency_distribution(month_draws)
            
            self.temporal_patterns = {
                'day_of_week': dow_patterns,
                'monthly': monthly_patterns
            }
        
        self._analyze_draw_intervals(draws_df)
        
        return self.temporal_patterns
    
    def analyze_positional_patterns(self, draws_df: pd.DataFrame) -> Dict:
        """Analyze which numbers appear in which positions"""
        logger.info("Analyzing positional patterns...")
        
        positional_freq = defaultdict(lambda: defaultdict(int))
        
        for _, draw in draws_df.iterrows():
            numbers = sorted(list(draw.values))
            for pos, number in enumerate(numbers):
                positional_freq[pos][number] += 1
        
        self.positional_patterns = {}
        for pos in range(OZLOTTO_NUMBERS):
            total = sum(positional_freq[pos].values())
            if total > 0:
                self.positional_patterns[pos] = {
                    num: count / total 
                    for num, count in positional_freq[pos].items()
                }
        
        return self.positional_patterns
    
    def build_correlation_matrix(self, draws_df: pd.DataFrame) -> np.ndarray:
        """Build correlation matrix for number co-occurrences"""
        logger.info("Building correlation matrix...")
        
        correlation_matrix = np.zeros((MAX_NUMBER, MAX_NUMBER))
        
        for _, draw in draws_df.iterrows():
            numbers = list(draw.values)
            for i, num1 in enumerate(numbers):
                for j, num2 in enumerate(numbers):
                    if i != j and 1 <= num1 <= MAX_NUMBER and 1 <= num2 <= MAX_NUMBER:
                        correlation_matrix[num1-1][num2-1] += 1
        
        total_draws = len(draws_df)
        if total_draws > 0:
            correlation_matrix = correlation_matrix / total_draws
        
        self.correlation_matrix = correlation_matrix
        return correlation_matrix
    
    def analyze_number_relationships(self, draws_df: pd.DataFrame) -> Dict:
        """Analyze relationships between numbers using network analysis"""
        logger.info("Analyzing number relationships...")
        
        G = nx.Graph()
        
        for num in range(MIN_NUMBER, MAX_NUMBER + 1):
            G.add_node(num)
        
        for _, draw in draws_df.iterrows():
            numbers = list(draw.values)
            for i, num1 in enumerate(numbers):
                for j, num2 in enumerate(numbers[i+1:], i+1):
                    if G.has_edge(num1, num2):
                        G[num1][num2]['weight'] += 1
                    else:
                        G.add_edge(num1, num2, weight=1)
        
        centrality = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)
        clustering = nx.clustering(G)
        
        return {
            'centrality': centrality,
            'betweenness': betweenness,
            'clustering': clustering,
            'graph': G
        }
    
    def detect_hot_cold_cycles(self, draws_df: pd.DataFrame, window_size: int = 20) -> Dict:
        """Detect hot and cold number cycles"""
        logger.info("Detecting hot/cold cycles...")
        
        hot_cold_data = {}
        
        for num in range(MIN_NUMBER, MAX_NUMBER + 1):
            appearances = []
            
            for i in range(len(draws_df) - window_size + 1):
                window = draws_df.iloc[i:i+window_size]
                count = (window.values == num).sum()
                appearances.append(count)
            
            if appearances:
                hot_cold_data[num] = {
                    'mean_frequency': np.mean(appearances),
                    'std_frequency': np.std(appearances),
                    'current_trend': self._calculate_trend(appearances[-10:]) if len(appearances) >= 10 else 0
                }
        
        return hot_cold_data
    
    def predict_based_on_patterns(self, draws_df: pd.DataFrame, n_predictions: int = 5) -> List[List[int]]:
        """Generate predictions based on pattern analysis"""
        logger.info("Generating pattern-based predictions...")
        
        predictions = []
        
        if not self.temporal_patterns:
            self.analyze_temporal_patterns(draws_df)
        if not self.positional_patterns:
            self.analyze_positional_patterns(draws_df)
        if self.correlation_matrix is None:
            self.build_correlation_matrix(draws_df)
        
        for _ in range(n_predictions):
            prediction = self._generate_pattern_prediction(draws_df)
            if prediction and len(set(prediction)) == OZLOTTO_NUMBERS:
                predictions.append(sorted(prediction))
        
        return predictions
    
    def _calculate_frequency_distribution(self, draws_subset: pd.DataFrame) -> Dict[int, float]:
        """Calculate frequency distribution for a subset of draws"""
        flat_numbers = draws_subset.values.flatten()
        total_numbers = len(flat_numbers)
        
        freq_dist = {}
        for num in range(MIN_NUMBER, MAX_NUMBER + 1):
            count = (flat_numbers == num).sum()
            freq_dist[num] = count / total_numbers if total_numbers > 0 else 0
        
        return freq_dist
    
    def _analyze_draw_intervals(self, draws_df: pd.DataFrame):
        """Analyze intervals between draws for each number"""
        number_intervals = defaultdict(list)
        
        for num in range(MIN_NUMBER, MAX_NUMBER + 1):
            last_appearance = -1
            
            for idx, (_, draw) in enumerate(draws_df.iterrows()):
                if num in draw.values:
                    if last_appearance >= 0:
                        interval = idx - last_appearance
                        number_intervals[num].append(interval)
                    last_appearance = idx
        
        interval_stats = {}
        for num, intervals in number_intervals.items():
            if intervals:
                interval_stats[num] = {
                    'mean_interval': np.mean(intervals),
                    'std_interval': np.std(intervals),
                    'median_interval': np.median(intervals)
                }
        
        self.temporal_patterns['intervals'] = interval_stats
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction (-1 to 1)"""
        if len(values) < 2:
            return 0
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return np.tanh(slope)
    
    def _generate_pattern_prediction(self, draws_df: pd.DataFrame) -> List[int]:
        """Generate a single prediction based on all patterns"""
        prediction = []
        
        for pos in range(OZLOTTO_NUMBERS):
            if pos in self.positional_patterns:
                candidates = list(self.positional_patterns[pos].keys())
                weights = list(self.positional_patterns[pos].values())
                
                if candidates and weights:
                    selected = np.random.choice(candidates, p=np.array(weights)/np.sum(weights))
                    if selected not in prediction:
                        prediction.append(selected)
        
        while len(prediction) < OZLOTTO_NUMBERS:
            if len(prediction) > 0 and self.correlation_matrix is not None:
                last_num = prediction[-1]
                correlations = self.correlation_matrix[last_num-1]
                
                candidates = []
                for num in range(MIN_NUMBER, MAX_NUMBER + 1):
                    if num not in prediction:
                        candidates.append((num, correlations[num-1]))
                
                if candidates:
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    top_candidates = candidates[:10]
                    selected = np.random.choice([c[0] for c in top_candidates])
                    prediction.append(selected)
                else:
                    num = np.random.randint(MIN_NUMBER, MAX_NUMBER + 1)
                    if num not in prediction:
                        prediction.append(num)
            else:
                num = np.random.randint(MIN_NUMBER, MAX_NUMBER + 1)
                if num not in prediction:
                    prediction.append(num)
        
        return prediction[:OZLOTTO_NUMBERS]
