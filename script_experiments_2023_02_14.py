from copy import copy, deepcopy
from irace2 import irace, dummy_stats_test
import itertools
import numpy as np
from sampling_functions import norm_sample, truncated_poisson, truncated_skellam
import scipy.stats as stats
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split,StratifiedShuffleSplit,cross_val_score
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import scipy.stats as ss
from scipy.stats import norm, poisson, skellam
from tqdm import tqdm
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

Xs = []
ys = []

df = pd.read_csv('spect_train.csv')
Xs.append(preprocessing.normalize(df.drop(columns=['OVERALL_DIAGNOSIS']).to_numpy()))
ys.append(df['OVERALL_DIAGNOSIS'].to_numpy())

df = pd.read_csv('spambase.csv')
Xs.append(preprocessing.normalize(df.drop(columns=['spam']).to_numpy()))
ys.append(df['spam'].to_numpy())

df = pd.read_csv('ionosphere_data.csv')
Xs.append(preprocessing.normalize(df.drop(columns=['column_ai']).to_numpy()))
ys.append(df['column_ai'].to_numpy())

#all the numeric parameters being configured must be set beforehand
models = [LogisticRegression(C=1), 
    RandomForestClassifier(n_estimators=100,max_depth=5,ccp_alpha=0.0),
    SVC(C=1,coef0=0.0),
    XGBClassifier(n_estimators=100,max_depth=6,subsample=1)]


parameters_dict = {
    'LogisticRegression': {'C': lambda loc : norm_sample(loc=loc, scale=1, min= 1e-2),
                            'penalty':['l2'],
                            'solver':['lbfgs','newton-cg','sag']},
    'SVC':{'C':lambda loc : norm_sample(loc=loc, scale=1, min= 1e-2),
            'coef0': lambda loc : norm_sample(loc=loc, scale=1, min= 1e-2),
            'kernel':['linear','poly','rbf','sigmoid'],
            'decision_function_shape':['ovo','ovr']},
    'RandomForestClassifier': {'n_estimators': lambda loc: truncated_skellam(loc, mu1=10, mu2=10, min=1), 
                                'max_depth': lambda loc: truncated_skellam(loc, mu1=1, mu2=1, min=1),
                                'max_features':['sqrt', 'log2', None],
                                'ccp_alpha':lambda loc : norm_sample(loc=loc, scale=0.1, min= 1e-3)
                                },
    'XGBClassifier': {'sample_type': ['uniform','weighted'], 
                        'max_depth': lambda loc: truncated_skellam(loc, mu1=1, mu2=1, min=1),
                        'booster':['gbtree','dart'],
                        'subsample':lambda loc : norm_sample(loc=loc, scale=0.3, min= 1e-2,max=1)}
}

stat_tests = [ ss.ttest_rel,
                ss.ttest_ind,
                ss.mannwhitneyu,
                ss.wilcoxon,
                dummy_stats_test]

data_set_id = [0,1,2]
cv_splits = [10, 30]
pop_size = [10, 50]
n_gen = [100]

factors = list(itertools.product(data_set_id,stat_tests,cv_splits,pop_size,n_gen))

n = 10
res = []

for n_exp in tqdm(range(n)):
    for f in tqdm(factors):
        data_id = f[0]
        X = Xs[data_id]
        y = ys[data_id]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

        stat_test = f[1]
        split = f[2]
        p_size = f[3]
        stop = f[4]

        best_model,best_scores,population,pop_scores = irace(models, 
                X_train, 
                y_train, 
                lambda x: x > stop, 
                stat_test, 
                parameters_dict, 
                p_size, 'f1', cv=split)

        best_model.fit(X_train,y_train)
        y_pred = best_model.predict(X_test)

        row = [data_id,stat_test,split,p_size,stop,'cv',type(best_model).__name__,f1_score(y_test,y_pred)]
        res.append(row)

        best_model,best_scores,population,pop_scores = irace(models, 
                X_train, 
                y_train, 
                lambda x: x > f[4], 
                stat_test, 
                parameters_dict, 
                p_size, 'f1', r=split)

        best_model.fit(X_train,y_train)
        y_pred = best_model.predict(X_test)

        row = [data_id,stat_test.__name__,split,p_size,stop,'train_test',type(best_model).__name__,f1_score(y_test,y_pred)]
        res.append(row)

        df = pd.DataFrame(res, columns = ['data_id', 'stat_test','n','p_size','max_iter','type','best_model','f1_score'])

        df.to_csv('results_2023-02-15.csv')
