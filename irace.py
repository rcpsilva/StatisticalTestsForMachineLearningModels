import numpy as np
import random
import scipy.stats as stats
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tqdm import tqdm
from sklearn import preprocessing
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import statsmodels.stats.weightstats as stats
import scipy.stats as ss
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit,cross_val_score

def irace(models, X, y, stop_condition, stat_test, parameters_dict):
    population = models
    generation = 0
    while not stop_condition(generation):
        print(f'Gen {generation}\n')
        new_population = []
        for i in range(len(population)):
            competitors = random.sample(population, 2)
            grid_searches = []
            for competitor in competitors:
                parameters = parameters_dict[type(competitor).__name__]
                grid_search = RandomizedSearchCV(competitor, parameters, scoring='f1', cv=10)
                grid_search.fit(X, y)
                grid_searches.append(grid_search)
            scores = [grid_search.cv_results_['mean_test_score'] for grid_search in grid_searches]
            best_scores = [grid_search.best_score_ for grid_search in grid_searches]
            t, p = stat_test(scores[0], scores[1]) #stats.ttest_rel(scores[0], scores[1])
            if p <= 0.05:
                best = grid_searches[np.argmin(-np.array(best_scores))]
            else:
                best = random.choice(grid_searches)
            new_population.append(best.best_estimator_)
        population = new_population
        generation += 1
    return population[0]


if __name__ == '__main__':

    df = pd.read_csv('spect_train.csv')
    X = preprocessing.normalize(df.drop(columns=['OVERALL_DIAGNOSIS']).to_numpy())
    y = df['OVERALL_DIAGNOSIS'].to_numpy()

    stop_condition = lambda generation: generation >= 10

    models = [LogisticRegression(), RandomForestClassifier(), XGBClassifier()]
   
    parameters_dict = {
        'LogisticRegression': {'C': [0.1, 0.5, 1, 10],'penalty':['l2'],'solver':['lbfgs','newton-cg','sag']},
        'RandomForestClassifier': {'n_estimators': [10, 50, 100, 200], 'max_depth': [None, 5, 10, 20]},
        'XGBClassifier': {'n_estimators': [5, 10, 50, 100], 'max_depth': [3, 5, 10, 20]}
    }

    # Possible tests
    # Mann-Whitney U Test: scipy.stats.mannwhitneyu
    # Wilcoxon Signed-Rank Test: scipy.stats.wilcoxon
    # Kruskal-Wallis H Test: scipy.stats.kruskal
    # ANOVA (Analysis of Variance): scipy.stats.f_oneway
    # Welch's t-test: scipy.stats.ttest_ind(a, b, equal_var=False)

    stat_test = ss.ttest_rel #stats.ttest_ind, stats.mannwhitneyu

    best_model = irace(models, X, y, lambda x: x > 100, stat_test, parameters_dict)

    
    print('============================================')
    print('============================================')
    print('============================================')
    print('============================================')
    print('============================================')
    print('============================================')
    print('============================================')
    print('============================================')    
    print('===============Baselines=============================')

    scores = cross_val_score(LogisticRegression(), X, y, cv=10, scoring='f1')
    
    print('LR')
    print(f'{np.mean(scores)} +- {np.std(scores)}')

    scores = cross_val_score(RandomForestClassifier(), X, y, cv=10, scoring='f1')
    
    print('RF')
    print(f'{np.mean(scores)} +- {np.std(scores)}')

    scores = cross_val_score(XGBClassifier(), X, y, cv=10, scoring='f1')
    
    print('XGB')
    print(f'{np.mean(scores)} +- {np.std(scores)}')

    scores = cross_val_score(best_model, X, y, cv=10, scoring='f1')

    print(best_model)
    print(f'{np.mean(scores)} +- {np.std(scores)}')
