import numpy as np
import random

def irace(population, evaluate_function, stop_condition):
    generation = 0
    while not stop_condition(generation):
        new_population = []
        for i in range(len(population)):
            competitors = random.sample(population, 2)
            best = min(competitors, key=evaluate_function)
            new_population.append(best)
        population = new_population
        generation += 1
    return population[0]

def evaluate_example(individual):
    return np.sum(np.abs(individual))

result = irace(np.random.rand(100, 10), evaluate_example, lambda generation: generation >= 100)
