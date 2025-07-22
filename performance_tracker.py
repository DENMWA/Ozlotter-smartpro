"""
Real-time performance monitoring and tracking
"""
import json
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
from config import *
from utils import setup_logger

logger = setup_logger(__name__)

class PerformanceTracker:
    def __init__(self):
        self.performance_file = f"{DATA_DIR}/performance_history.json"
        self.session_data = []
        
    def track_prediction_performance(self, predictions: List[List[int]], 
                                   method_used: str,
                                   generation_info: Optional[Dict] = None) -> str:
        """Track performance of a prediction batch"""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        session_record = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'method_used': method_used,
            'predictions_count': len(predictions),
            'predictions': predictions,
            'generation_info': generation_info or {},
            'performance_metrics': self._calculate_prediction_metrics(predictions)
        }
        
        self.session_data.append(session_record)
        self._save_performance_data()
        
        logger.info(f"Tracked performance for session {session_id}")
        return session_id
    
    def update_with_actual_results(self, session_id: str, actual_draw: List[int]) -> Dict:
        """Update session with actual lottery results"""
        for session in self.session_data:
            if session['session_id'] == session_id:
                results = self._evaluate_against_actual(session['predictions'], actual_draw)
                session['actual_results'] = {
                    'actual_draw': actual_draw,
                    'evaluation_date': datetime.now().isoformat(),
                    'results': results
                }
                
                self._save_performance_data()
                logger.info(f"Updated session {session_id} with actual results")
                return results
        
        logger.warning(f"Session {session_id} not found")
        return {}
    
    def get_performance_summary(self, last_n_sessions: int = 10) -> Dict:
        """Get performance summary for recent sessions"""
        if not self.session_data:
            self._load_performance_data()
        
        recent_sessions = self.session_data[-last_n_sessions:] if self.session_data else []
        
        if not recent_sessions:
            return {"status": "no_data"}
        
        evaluated_sessions = [s for s in recent_sessions if 'actual_results' in s]
        
        if not evaluated_sessions:
            return {
                "status": "no_evaluated_sessions",
                "total_sessions": len(recent_sessions),
                "pending_evaluation": len(recent_sessions)
            }
        
        all_matches = []
        method_performance = {}
        
        for session in evaluated_sessions:
            results = session['actual_results']['results']
            method = session['method_used']
            
            if method not in method_performance:
                method_performance[method] = []
            
            best_match = results.get('best_match', 0)
            all_matches.append(best_match)
            method_performance[method].append(best_match)
        
        summary = {
            "status": "success",
            "total_sessions": len(recent_sessions),
            "evaluated_sessions": len(evaluated_sessions),
            "pending_sessions": len(recent_sessions) - len(evaluated_sessions),
            "overall_performance": {
                "average_best_match": np.mean(all_matches) if all_matches else 0,
                "max_match_achieved": max(all_matches) if all_matches else 0,
                "sessions_with_3plus_matches": sum(1 for m in all_matches if m >= 3),
                "hit_rate_3plus": sum(1 for m in all_matches if m >= 3) / len(all_matches) * 100 if all_matches else 0
            },
            "method_performance": {
                method: {
                    "sessions": len(matches),
                    "average_match": np.mean(matches),
                    "best_match": max(matches)
                }
                for method, matches in method_performance.items()
            }
        }
        
        return summary
    
    def _calculate_prediction_metrics(self, predictions: List[List[int]]) -> Dict:
        """Calculate metrics for a set of predictions"""
        all_numbers = [num for pred in predictions for num in pred]
        
        return {
            'diversity_score': self._calculate_diversity(predictions),
            'number_distribution': {
                'unique_numbers': len(set(all_numbers)),
                'most_common': pd.Series(all_numbers).value_counts().head(5).to_dict(),
                'range_distribution': self._analyze_range_distribution(all_numbers)
            },
            'statistical_properties': {
                'average_sum': np.mean([sum(pred) for pred in predictions]),
                'sum_variance': np.var([sum(pred) for pred in predictions]),
                'even_odd_ratio': np.mean([sum(1 for x in pred if x % 2 == 0) / len(pred) for pred in predictions])
            }
        }
    
    def _calculate_diversity(self, predictions: List[List[int]]) -> float:
        """Calculate diversity score for predictions"""
        if len(predictions) < 2:
            return 1.0
        
        total_similarity = 0
        pair_count = 0
        
        for i, pred1 in enumerate(predictions):
            for pred2 in predictions[i+1:]:
                similarity = len(set(pred1) & set(pred2)) / OZLOTTO_NUMBERS
                total_similarity += similarity
                pair_count += 1
        
        avg_similarity = total_similarity / pair_count if pair_count > 0 else 0
        return 1.0 - avg_similarity
    
    def _analyze_range_distribution(self, numbers: List[int]) -> Dict:
        """Analyze distribution across number ranges"""
        low = sum(1 for n in numbers if 1 <= n <= 15)
        mid = sum(1 for n in numbers if 16 <= n <= 31)
        high = sum(1 for n in numbers if 32 <= n <= 47)
        total = len(numbers)
        
        return {
            'low_range_pct': low / total * 100 if total > 0 else 0,
            'mid_range_pct': mid / total * 100 if total > 0 else 0,
            'high_range_pct': high / total * 100 if total > 0 else 0
        }
    
    def _evaluate_against_actual(self, predictions: List[List[int]], actual_draw: List[int]) -> Dict:
        """Evaluate predictions against actual lottery draw"""
        matches = []
        detailed_results = []
        
        for i, prediction in enumerate(predictions):
            match_count = len(set(prediction) & set(actual_draw))
            matches.append(match_count)
            
            detailed_results.append({
                'prediction_id': i + 1,
                'prediction': prediction,
                'matches': match_count,
                'matched_numbers': list(set(prediction) & set(actual_draw))
            })
        
        return {
            'best_match': max(matches) if matches else 0,
            'average_match': np.mean(matches) if matches else 0,
            'total_predictions': len(predictions),
            'predictions_with_3plus': sum(1 for m in matches if m >= 3),
            'predictions_with_4plus': sum(1 for m in matches if m >= 4),
            'predictions_with_5plus': sum(1 for m in matches if m >= 5),
            'detailed_results': detailed_results
        }
    
    def _save_performance_data(self):
        """Save performance data to file"""
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(self.performance_file, 'w') as f:
                json.dump(self.session_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving performance data: {e}")
    
    def _load_performance_data(self):
        """Load performance data from file"""
        try:
            if os.path.exists(self.performance_file):
                with open(self.performance_file, 'r') as f:
                    self.session_data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading performance data: {e}")
            self.session_data = []
