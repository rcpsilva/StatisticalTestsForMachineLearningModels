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
from sklearn.model_selection import cross_val_score, permutation_test_score
from copy import copy, deepcopy
from sampling_functions import truncated_poisson, truncated_skellam, norm_sample
from validation_functions import repeated_train_test


def irace2(models, X, y, stop_condition, stat_test, parameters_dict, scoring, 
           cv=None, r=100, show_gen=False, p_value=0.05):
    ''' Irace finds a population of models that maximizes the score given by the scoring function.
    
    '''
    # create an empty dictionary to hold the population. 
    # the population stores the best hyperparameter found for each model
    population = {}

    # loop through each model in the models list
    for model in models:
        # get the parameter names for the current model from the parameters_dict
        par_names = parameters_dict[type(model).__name__]
        # create a dictionary of the parameter names and their values using the getattr function
        par_val = {name: getattr(model, name) for name in par_names}
        # add the parameter dictionary to the population dictionary with the model name as the key
        population.update({type(model).__name__:dict(par_val)})

    generation = 0
    
    if cv:
        pop_scores = {type(model).__name__:cross_val_score(model, X, y, cv=cv, scoring=scoring) for model in models}
    else:
        pop_scores = {type(model).__name__:repeated_train_test(model, X, y, n=r, scoring=scoring) for model in models}

    avg_scores = [np.mean(scores) for scores in list(pop_scores.values())]
    m_names = list(pop_scores.keys())
    
    best_model = m_names[np.argmax(avg_scores)]
    best_scores = copy(pop_scores[best_model])

    initial_best = copy(best_scores)

    initial_scores = copy(pop_scores)

    while not stop_condition(generation):
        if show_gen:
            print(f'Gen {generation}\n')

        # Select competitor
        competitor = copy(random.sample(models, 1)[0])
        # Get the the best parameters ever found for the competitor model
        parameters = population[type(competitor).__name__]        
        # Get the list of parameter of the model. Used for variation
        parameter_list = parameters_dict[type(competitor).__name__]

        # Vary parameters
        for p in parameters:   
            if isinstance(parameter_list[p],list):
                setattr(competitor,p,random.sample(parameter_list[p], 1)[0]) 
            else:
                setattr(competitor,p,parameter_list[p](parameters[p]))

        # Compute scores
        if cv:
            scores = cross_val_score(competitor, X, y, cv=cv, scoring=scoring)
        else:
            scores = repeated_train_test(competitor, X, y, n=r, scoring=scoring)

        # Check if it is an improvement over the best model of the same type
        t, p = stat_test(pop_scores[type(competitor).__name__], scores) 

        if p <= p_value and np.mean(scores) > np.mean(pop_scores[type(competitor).__name__]):  
                
                # Update best paramters
                # get the parameter names for the current model from the parameters_dict
                par_names = parameters_dict[type(competitor).__name__]
                # create a dictionary of the parameter names and their values using the getattr function
                par_val = {name: getattr(competitor, name) for name in par_names}
                
                population[type(competitor).__name__] = copy(par_val)
                pop_scores[type(competitor).__name__] = copy(scores)

                # Check if it is an improvement over the best model
                t, p = stat_test(best_scores, scores) 
                if p <= p_value and np.mean(scores) > np.mean(best_scores):
                    best_model = type(competitor).__name__
                    best_scores = copy(scores)
        generation += 1
        if show_gen:
            print(f'Average scores: {np.mean([np.mean(pop_scores[model]) for model in pop_scores]):.4f}')
            print(f'Best average score: {np.mean(best_scores):.4f}')
    return best_model,best_scores,population,pop_scores,initial_scores,initial_best

def irace(models, X, y, stop_condition, stat_test, parameters_dict, pop_size, scoring, cv=None, r=100, show_gen=False):
    ''' Irace finds a population of models that maximizes the score given by the scoring function.
    
    '''
    population = [copy(r) for r in random.choices(models, k=pop_size)]
    generation = 0
    
    if cv:
        pop_scores = [cross_val_score(model, X, y, cv=cv, scoring=scoring) for model in population]
    else:
        pop_scores = [repeated_train_test(model, X, y, n=r, scoring=scoring) for model in population]

    best_model = copy(population[0])
    best_scores = copy(pop_scores[0])

    while not stop_condition(generation):
        if show_gen:
            print(f'Gen {generation}\n')
        for i in range(len(population)):
            # Select competitor
            competitor = copy(random.sample(population, 1)[0])
            # Get the specific parameters for the competitor model
            parameters = parameters_dict[type(competitor).__name__]
            # Vary parameters
            for p in parameters:   
                if isinstance(parameters[p],list):
                    setattr(competitor,p,random.sample(parameters[p], 1)[0]) 
                else:
                    setattr(competitor,p,parameters[p](getattr(competitor,p)))

            if cv:
                scores = cross_val_score(competitor, X, y, cv=cv, scoring=scoring)
            else:
                scores = repeated_train_test(competitor, X, y, n=r, scoring=scoring)

            t, p = stat_test(pop_scores[i], scores) 

            if p <= 0.05 and np.mean(scores) > np.mean(pop_scores[i]):  
                    population[i] = competitor
                    pop_scores[i] = scores
                    t, p = stat_test(best_scores, scores) 
                    if p <= 0.05 and np.mean(scores) > np.mean(best_scores):
                        best_model = copy(competitor)
                        best_scores = copy(scores)
        generation += 1
        if show_gen:
            print(f'Average scores: {np.mean([np.mean(scores) for scores in pop_scores])}')
    return best_model,best_scores,population,pop_scores

def dummy_stats_test(a,b):
    return 0,0

if __name__ == '__main__':

    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    import warnings
    warnings.filterwarnings("ignore")


    df = pd.read_csv('UCI_Credit_Card.csv')
    X = (preprocessing.normalize(df.drop(columns=['ID','default.payment.next.month']).to_numpy()))
    y = (df['default.payment.next.month'].to_numpy())

    #all the parameters being configures must be set beforehand
    models = [LogisticRegression(C=1), 
        #RandomForestClassifier(n_estimators=100,max_depth=5),
        KNeighborsClassifier(n_neighbors=5),
        DecisionTreeClassifier(max_depth=5)
        #SVC(C=1,coef0=0.0),
        #XGBClassifier(n_estimators=100,max_depth=6,subsample=1)
        ]
   

    parameters_dict = {
        'LogisticRegression': {'C': lambda loc : norm_sample(loc=loc, scale=2, min= 1e-2),
                                'penalty':['l2'],
                                'solver':['lbfgs','newton-cg','sag']},
        'KNeighborsClassifier':{'n_neighbors':lambda loc: truncated_skellam(loc, mu1=2, mu2=2, min=3),
                                'weights':['uniform', 'distance']},
        'DecisionTreeClassifier':{'max_depth':lambda loc: truncated_skellam(loc, mu1=2, mu2=2, min=2),
                                  'max_features':['sqrt','log2',None],
                                  'criterion':['gini','entropy','log_loss']},                        
        #'SVC':{'C':lambda loc : norm_sample(loc=loc, scale=1, min= 1e-2),
        #        'coef0': lambda loc : norm_sample(loc=loc, scale=1, min= 1e-2),
        #        'kernel':['linear','poly','rbf','sigmoid'],
        #        'decision_function_shape':['ovo','ovr']},
        #'RandomForestClassifier': {'n_estimators': lambda loc: truncated_skellam(loc, mu1=10, mu2=10, min=3), 
        #                            'max_depth': lambda loc: truncated_skellam(loc, mu1=2, mu2=2, min=2),
        #                            'max_features':['sqrt', 'log2', None]
        #                            },
        #'XGBClassifier': {'tree_method': ['auto','exact','approx'], 
        #                    'max_depth': lambda loc: truncated_skellam(loc, mu1=1, mu2=1, min=1),
        #                    'booster':['gbtree','dart'],
        #                    'subsample':lambda loc : norm_sample(loc=loc, scale=0.3, min= 1e-2,max=1)}
    }

    # Possible tests
    # Mann-Whitney U Test: scipy.stats.mannwhitneyu
    # Wilcoxon Signed-Rank Test: scipy.stats.wilcoxon
    # Kruskal-Wallis H Test: scipy.stats.kruskal
    # ANOVA (Analysis of Variance): scipy.stats.f_oneway
    # Welch's t-test: scipy.stats.ttest_ind(a, b, equal_var=False)

    stat_test = ss.ttest_rel #stats.ttest_ind, stats.mannwhitneyu

    best_model,best_scores,population,pop_scores,initial_scores,initial_best = irace2(models, 
                                                                         X, 
                                                                         y, 
                                                                         lambda x: x > 500, 
                                                                         stat_test, 
                                                                         parameters_dict, 
                                                                         scoring='f1',
                                                                         r=50,
                                                                         show_gen=True)

    
    avg_scores = np.mean([np.mean(scores) for scores in list(pop_scores.values())])
    initial_scores = np.mean([np.mean(scores) for scores in list(initial_scores.values())])

    print(f'Ini_best: {np.mean(initial_best):.4f}')
    print(f'Best: {np.mean(best_scores):.4f}')
    print(f'Initial: {initial_scores:.4f}')
    print(f'Final: {avg_scores:.4f}')
    