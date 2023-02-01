import numpy as np
import random
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

def irace(X, y, stop_condition, parameters):
    population = [xgb.XGBRegressor()]
    generation = 0
    while not stop_condition(generation):
        new_population = []
        for i in range(len(population)):
            competitors = random.sample(population, 2)
            grid_searches = [GridSearchCV(competitor, parameters, scoring='neg_mean_absolute_error', cv=5) for competitor in competitors]
            for grid_search in grid_searches:
                grid_search.fit(X, y)
            best = min(grid_searches, key=lambda grid_search: grid_search.best_score_)
            new_population.append(best.best_estimator_)
        population = new_population
        generation += 1
    return population[0]

parameters = {'n_estimators': [100, 200, 300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [3, 5, 7]}

result = irace(X, y, lambda generation: generation >= 100, parameters)