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
from scipy.stats import norm, poisson, skellam
from copy import copy, deepcopy

def irace(models, X, y, stop_condition, stat_test, parameters_dict, pop_size = 10, cv = 5, scoring='f1_macro'):
    ''' Irace finds a population of models that maximizes the score given by the scoring function.
    
    '''
    population = [deepcopy(r) for r in random.choices(models, k=pop_size)]
    generation = 0
    pop_scores = [cross_val_score(model, X, y, cv=cv, scoring=scoring) for model in population]

    while not stop_condition(generation):
        print(f'Gen {generation}\n')
        for i in range(len(population)):
            # Select competitor
            competitor = deepcopy(random.sample(population, 1)[0])
            # Get the specific parameters for the competitor model
            parameters = parameters_dict[type(competitor).__name__]
            # Vary parameters
            for p in parameters:   
                if isinstance(parameters[p],list):
                    setattr(competitor,p,random.sample(parameters[p], 1)[0]) 
                else:
                    setattr(competitor,p,parameters[p](getattr(competitor,p)))

            scores = cross_val_score(competitor, X, y, cv=cv, scoring=scoring)

            t, p = stat_test(pop_scores[i], scores) #stats.ttest_rel(scores[0], scores[1])
            if p <= 0.05:
                if np.mean(scores) > np.mean(pop_scores[i]):  
                    population[i] = competitor
                    pop_scores[i] = scores
            else:
                population[i] = competitor
                pop_scores[i] = scores
        generation += 1
    return population, pop_scores


def norm_sample(loc, scale, min):
    dist = norm(loc=loc, scale=scale)
    sample = dist.rvs(1)[0] 
    return sample if sample > min else min 

def truncated_poisson(loc, mu, min):
    #dist = poisson(mu=mu)
    sample = poisson.rvs(mu=mu, loc=loc, size=1)[0] 
    return sample if sample > min else min 

def truncated_skellam(loc, mu1, mu2, min):
    sample = skellam.rvs(mu1=mu1, mu2=mu2, loc=loc, size=1)[0] 
    return sample if sample > min else min 



if __name__ == '__main__':

    df = pd.read_csv('spect_train.csv')
    X = preprocessing.normalize(df.drop(columns=['OVERALL_DIAGNOSIS']).to_numpy())
    y = df['OVERALL_DIAGNOSIS'].to_numpy()

    stop_condition = lambda generation: generation >= 10

    #all the parameters being configures must be set beforehand
    models = [LogisticRegression(C=1), 
        RandomForestClassifier(n_estimators=100,max_depth=5),
        XGBClassifier(n_estimators=100,max_depth=6)]
   

    parameters_dict = {
        'LogisticRegression': {'C': lambda loc : norm_sample(loc=loc, scale=1, min= 1e-2),
                                'penalty':['l2'],
                                'solver':['lbfgs','newton-cg','sag']},
        'RandomForestClassifier': {'n_estimators': lambda loc: truncated_skellam(loc, mu1=10, mu2=10, min=1), 
                                    'max_depth': lambda loc: truncated_skellam(loc, mu1=1, mu2=1, min=1)},
        'XGBClassifier': {'sample_type': ['uniform','weighted'], 
                            'max_depth': lambda loc: truncated_skellam(loc, mu1=1, mu2=1, min=1)}
    }

    # Possible tests
    # Mann-Whitney U Test: scipy.stats.mannwhitneyu
    # Wilcoxon Signed-Rank Test: scipy.stats.wilcoxon
    # Kruskal-Wallis H Test: scipy.stats.kruskal
    # ANOVA (Analysis of Variance): scipy.stats.f_oneway
    # Welch's t-test: scipy.stats.ttest_ind(a, b, equal_var=False)

    stat_test = ss.ttest_rel #stats.ttest_ind, stats.mannwhitneyu

    pop, pop_scores = irace(models, X, y, lambda x: x > 500, stat_test, parameters_dict, pop_size = 15, cv = 10, scoring='f1_macro')

    scores = cross_val_score(LogisticRegression(), X, y, cv=10, scoring='f1')
    print('LR')
    print(f'{np.mean(scores)} +- {np.std(scores)}')

    scores = cross_val_score(RandomForestClassifier(), X, y, cv=10, scoring='f1')    
    print('RF')
    print(f'{np.mean(scores)} +- {np.std(scores)}')


    print()
    for i in range(len(pop)):
        print(pop[i])
        print(f'{np.mean(pop_scores[i])} +- {np.std(pop_scores[i])}')
