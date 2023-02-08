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
from scipy.stats import norm
from scipy.stats import poisson

def irace(models, X, y, stop_condition, stat_test, parameters_dict, cv = 5, scoring='f1_macro'):
    population = models
    generation = 0

    pop_scores = [cross_val_score(model, X, y, cv=cv, scoring=scoring) for model in population]

    while not stop_condition(generation):
        print(f'Gen {generation}\n')
        for i in range(len(population)):
            # Select competitor
            competitor = random.sample(population, 1)[0]
            # Get the specific parameters for the competitor model
            parameters = parameters_dict[type(competitor).__name__]
            # Vary parameters
            for p in parameters:   
                if isinstance(parameters[p],list):
                    setattr(competitor,p,random.sample(parameters[p], 1)) 
                else:
                    dist = parameters[p](getattr(competitor,p))
                    setattr(competitor,p,dist.rvs(1))

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
    return population


if __name__ == '__main__':

    df = pd.read_csv('spect_train.csv')
    X = preprocessing.normalize(df.drop(columns=['OVERALL_DIAGNOSIS']).to_numpy())
    y = df['OVERALL_DIAGNOSIS'].to_numpy()

    stop_condition = lambda generation: generation >= 10

    models = [LogisticRegression(), RandomForestClassifier()]
   
    parameters_dict = {
        'LogisticRegression': {'C': lambda loc : norm(loc=loc, scale=1),'penalty':['l2'],'solver':['lbfgs','newton-cg','sag']},
        'XGBClassifier': {'n_estimators': lambda mu: poisson(mu=mu), 'max_depth': lambda mu: poisson(mu=mu)}
    }

    # Possible tests
    # Mann-Whitney U Test: scipy.stats.mannwhitneyu
    # Wilcoxon Signed-Rank Test: scipy.stats.wilcoxon
    # Kruskal-Wallis H Test: scipy.stats.kruskal
    # ANOVA (Analysis of Variance): scipy.stats.f_oneway
    # Welch's t-test: scipy.stats.ttest_ind(a, b, equal_var=False)

    stat_test = ss.ttest_rel #stats.ttest_ind, stats.mannwhitneyu

    best_model = irace(models, X, y, lambda x: x > 100, stat_test, parameters_dict, cv = 10, scoring='f1_macro')

    
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
    
    #print('XGB')
    #print(f'{np.mean(scores)} +- {np.std(scores)}')

    print('best')
    scores = cross_val_score(best_model, X, y, cv=10, scoring='f1')
    print(best_model)
    print(f'{np.mean(scores)} +- {np.std(scores)}')
