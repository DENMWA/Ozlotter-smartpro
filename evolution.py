
import random

def crossover(parent1, parent2):
    """Performs crossover between two parents to create a base child set"""
    child = list(set(random.sample(parent1, 4) + random.sample(parent2, 3)))
    while len(child) < 7:
        child.append(random.randint(1, 47))
    return sorted(list(set(child)))[:7]

def mutate(base_set, mutation_pool, n=1):
    """Mutates a base set using a pool of high-value numbers (e.g. from past Div1/2 hits)"""
    child = base_set[:]
    for _ in range(n):
        idx = random.randint(0, 6)
        new_val = random.choice(mutation_pool)
        while new_val in child:
            new_val = random.choice(mutation_pool)
        child[idx] = new_val
    return sorted(child)

def generate_generation(seed_sets, mutation_pool, n_children=100, mutation_strength=1):
    """Creates a full generation of n_children sets using crossover and mutation"""
    generation = []
    while len(generation) < n_children:
        p1, p2 = random.sample(seed_sets, 2)
        base = crossover(p1, p2)
        child = mutate(base, mutation_pool, n=mutation_strength)
        if len(set(child)) == 7 and all(1 <= x <= 47 for x in child):
            generation.append(child)
    return generation
