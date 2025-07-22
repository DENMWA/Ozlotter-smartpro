"""
Advanced genetic evolution with multiple strategies and adaptive parameters
"""
import random
import numpy as np
from typing import List, Dict, Tuple
from config import *
from utils import setup_logger

logger = setup_logger(__name__)

class AdaptiveEvolutionEngine:
    def __init__(self):
        self.mutation_rate = DEFAULT_MUTATION_RATE
        self.performance_history = []
        self.strategy_weights = {
            'frequency': 0.25,
            'entropy': 0.25,
            'gap': 0.25,
            'correlation': 0.25
        }
    
    def quantum_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Quantum-inspired crossover using superposition principles"""
        combined_pool = list(set(parent1 + parent2))
        
        child = []
        while len(child) < OZLOTTO_NUMBERS:
            for num in combined_pool:
                if len(child) >= OZLOTTO_NUMBERS:
                    break
                
                prob = 0.3  # base probability
                if num in parent1:
                    prob += 0.3
                if num in parent2:
                    prob += 0.3
                
                if random.random() < prob and num not in child:
                    child.append(num)
        
        while len(child) < OZLOTTO_NUMBERS:
            num = random.randint(MIN_NUMBER, MAX_NUMBER)
            if num not in child:
                child.append(num)
        
        return sorted(child[:OZLOTTO_NUMBERS])
    
    def adaptive_mutation(self, base_set: List[int], mutation_pool: List[int], 
                         generation_performance: float = 0.5) -> List[int]:
        """Adaptive mutation with dynamic rate based on performance"""
        if generation_performance < 0.3:
            current_rate = min(self.mutation_rate * 1.5, 0.3)
        elif generation_performance > 0.7:
            current_rate = max(self.mutation_rate * 0.7, 0.05)
        else:
            current_rate = self.mutation_rate
        
        child = base_set[:]
        mutations = max(1, int(current_rate * OZLOTTO_NUMBERS))
        
        for _ in range(mutations):
            if random.random() < current_rate:
                idx = random.randint(0, OZLOTTO_NUMBERS - 1)
                new_val = random.choice(mutation_pool)
                
                attempts = 0
                while new_val in child and attempts < 20:
                    new_val = random.choice(mutation_pool)
                    attempts += 1
                
                if new_val not in child:
                    child[idx] = new_val
        
        return sorted(child)
    
    def island_model_evolution(self, seed_sets: List[List[int]], 
                              mutation_pool: List[int], 
                              n_children: int = 100) -> List[List[int]]:
        """Multi-population evolution with island model"""
        populations = {
            'frequency_focused': [],
            'entropy_focused': [],
            'gap_focused': [],
            'hybrid': []
        }
        
        children_per_pop = n_children // 4
        
        for pop_name in populations.keys():
            population = []
            for _ in range(children_per_pop):
                p1, p2 = random.sample(seed_sets, 2)
                
                if pop_name == 'frequency_focused':
                    child = self.frequency_focused_crossover(p1, p2, mutation_pool)
                elif pop_name == 'entropy_focused':
                    child = self.entropy_focused_crossover(p1, p2)
                elif pop_name == 'gap_focused':
                    child = self.gap_focused_crossover(p1, p2)
                else:
                    child = self.quantum_crossover(p1, p2)
                
                child = self.adaptive_mutation(child, mutation_pool)
                
                if self.validate_child(child):
                    population.append(child)
            
            populations[pop_name] = population
        
        self.migrate_best_individuals(populations)
        
        all_children = []
        for pop in populations.values():
            all_children.extend(pop)
        
        return all_children[:n_children]
    
    def frequency_focused_crossover(self, parent1: List[int], parent2: List[int], 
                                   mutation_pool: List[int]) -> List[int]:
        """Crossover focusing on high-frequency numbers"""
        child = []
        
        pool_numbers = random.sample(mutation_pool, min(4, len(mutation_pool)))
        child.extend([n for n in pool_numbers if n not in child])
        
        combined = parent1 + parent2
        for num in combined:
            if len(child) >= OZLOTTO_NUMBERS:
                break
            if num not in child:
                child.append(num)
        
        while len(child) < OZLOTTO_NUMBERS:
            num = random.randint(MIN_NUMBER, MAX_NUMBER)
            if num not in child:
                child.append(num)
        
        return sorted(child[:OZLOTTO_NUMBERS])
    
    def entropy_focused_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Crossover focusing on entropy optimization"""
        child = []
        
        ranges = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 47)]
        
        for i, (start, end) in enumerate(ranges):
            if len(child) >= OZLOTTO_NUMBERS:
                break
            
            candidates = [n for n in parent1 + parent2 if start <= n <= end and n not in child]
            if candidates:
                child.append(random.choice(candidates))
            else:
                num = random.randint(start, min(end, MAX_NUMBER))
                if num not in child:
                    child.append(num)
        
        while len(child) < OZLOTTO_NUMBERS:
            num = random.randint(MIN_NUMBER, MAX_NUMBER)
            if num not in child:
                child.append(num)
        
        return sorted(child[:OZLOTTO_NUMBERS])
    
    def gap_focused_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Crossover focusing on optimal gap distribution"""
        ideal_gap = MAX_NUMBER / OZLOTTO_NUMBERS
        child = []
        
        child.append(random.choice(parent1 + parent2))
        
        for i in range(1, OZLOTTO_NUMBERS):
            target = int(child[-1] + ideal_gap)
            
            candidates = parent1 + parent2
            closest = min(candidates, key=lambda x: abs(x - target))
            
            if closest not in child:
                child.append(closest)
            else:
                for offset in range(1, 10):
                    for direction in [-1, 1]:
                        candidate = target + (offset * direction)
                        if MIN_NUMBER <= candidate <= MAX_NUMBER and candidate not in child:
                            child.append(candidate)
                            break
                    if len(child) > i:
                        break
        
        while len(child) < OZLOTTER_NUMBERS:
            num = random.randint(MIN_NUMBER, MAX_NUMBER)
            if num not in child:
                child.append(num)
        
        return sorted(child[:OZLOTTER_NUMBERS])
    
    def migrate_best_individuals(self, populations: Dict[str, List[List[int]]]):
        """Migrate best individuals between populations"""
        migration_rate = 0.1
        
        for pop_name, population in populations.items():
            if len(population) < 2:
                continue
                
            n_migrants = max(1, int(len(population) * migration_rate))
            migrants = random.sample(population, n_migrants)
            
            other_pops = [p for p in populations.keys() if p != pop_name]
            if other_pops:
                target_pop = random.choice(other_pops)
                populations[target_pop].extend(migrants)
    
    def validate_child(self, child: List[int]) -> bool:
        """Validate that child meets lottery requirements"""
        return (len(set(child)) == OZLOTTER_NUMBERS and 
                all(MIN_NUMBER <= x <= MAX_NUMBER for x in child))
    
    def update_performance(self, performance_score: float):
        """Update performance history for adaptive behavior"""
        self.performance_history.append(performance_score)
        
        if len(self.performance_history) > 10:
            self.performance_history = self.performance_history[-10:]
        
        if len(self.performance_history) >= 3:
            recent_avg = np.mean(self.performance_history[-3:])
            if recent_avg < 0.3:
                self.mutation_rate = min(self.mutation_rate * 1.1, 0.3)
            elif recent_avg > 0.7:
                self.mutation_rate = max(self.mutation_rate * 0.9, 0.05)

def generate_advanced_generation(seed_sets: List[List[int]], 
                               mutation_pool: List[int], 
                               n_children: int = 100, 
                               mutation_strength: int = 1) -> List[List[int]]:
    """Enhanced generation function using advanced evolution"""
    engine = AdaptiveEvolutionEngine()
    return engine.island_model_evolution(seed_sets, mutation_pool, n_children)
