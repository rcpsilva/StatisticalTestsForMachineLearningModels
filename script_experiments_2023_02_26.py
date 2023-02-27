from copy import copy, deepcopy
from irace2 import irace2, dummy_stats_test
import itertools
import numpy as np
from sampling_functions import norm_sample, truncated_poisson, truncated_skellam
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split,StratifiedShuffleSplit,cross_val_score
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import scipy.stats as ss
from scipy.stats import norm, poisson, skellam
from xgboost import XGBClassifier
from tqdm import tqdm
import pandas as pd
import pickle
import sys
import warnings
import json
from json import JSONEncoder
warnings.filterwarnings("ignore")

# Get the string parameter from the command-line argument
csv_file = sys.argv[1]
json_file = sys.argv[2]


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


show_gen = False

Xs = []
ys = []

#df = pd.read_csv('spect_train.csv')
#Xs.append(preprocessing.normalize(df.drop(columns=['OVERALL_DIAGNOSIS']).to_numpy()))
#s.append(df['OVERALL_DIAGNOSIS'].to_numpy())

df = pd.read_csv('spambase.csv')
Xs.append(preprocessing.normalize(df.drop(columns=['spam']).to_numpy()))
ys.append(df['spam'].to_numpy())

#df = pd.read_csv('ionosphere_data.csv')
#Xs.append(preprocessing.normalize(df.drop(columns=['column_ai']).to_numpy()))
#ys.append(df['column_ai'].to_numpy())

df = pd.read_csv('heart.csv')
Xs.append(preprocessing.normalize(df.drop(columns=['target']).to_numpy()))
ys.append(df['target'].to_numpy())

df = pd.read_csv('UCI_Credit_Card.csv')
Xs.append(preprocessing.normalize(df.drop(columns=['ID','default.payment.next.month']).to_numpy()))
ys.append(df['default.payment.next.month'].to_numpy())

#all the parameters being configures must be set beforehand
models = [#LogisticRegression(C=1,solver='sag'), 
        #RandomForestClassifier(n_estimators=10,max_depth=1,max_features=None),
        KNeighborsClassifier(n_neighbors=3,weights='uniform'),
        DecisionTreeClassifier(max_depth=8,max_features=None,criterion='log_loss'),
        #SVC(C=1,coef0=0.0,decision_function_shape='ovo',kernel='linear'),
        XGBClassifier(n_estimators=1,max_depth=6,subsample=1)
        ]
   

parameters_dict = {
        #'LogisticRegression': {'C': lambda loc : norm_sample(loc=loc, scale=1, min= 1e-2),
        #                        'penalty':['l2'],
        #                        'solver':['lbfgs','newton-cg','sag']},
        'KNeighborsClassifier':{'n_neighbors':lambda loc: truncated_skellam(loc, mu1=1, mu2=1, min=3),
                                'weights':['uniform', 'distance']},
        'DecisionTreeClassifier':{'max_depth':lambda loc: truncated_skellam(loc, mu1=1, mu2=1, min=2),
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
        'XGBClassifier': {'tree_method': ['auto','exact','approx'], 
                            'max_depth': lambda loc: truncated_skellam(loc, mu1=1, mu2=1, min=1),
                            'booster':['gbtree','dart'],
                            'n_estimators': lambda loc: truncated_skellam(loc, mu1=1, mu2=1, min=1),
                            'subsample':lambda loc : norm_sample(loc=loc, scale=0.3, min= 1e-2,max=1)}
    }

stat_tests = [ ss.ttest_rel,
                ss.ttest_ind,
                ss.mannwhitneyu,
                dummy_stats_test]

data_set_id = np.arange(len(ys))
cv_splits = [10, 30, 50]
n_gen = [25,100]

factors = list(itertools.product(n_gen,cv_splits,data_set_id,stat_tests))

n = 10
res = []

exp_info = {'data_id':0,
            'stat_test':'',
            'split':0,
            'stop':0,
            'type_partition':'',
            'best_model_name':'',
            'test_score':'0',
            'scores_evolution':[],
            'best_evolution':[]}

exp_list = [copy(exp_info) for i in range(n*len(factors))] 
exp_number = 0

for n_exp in tqdm(range(n)):
    for f in tqdm(factors):
        data_id = f[2]
        X = Xs[data_id]
        y = ys[data_id]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

        stat_test = f[3]
        split = f[1]
        stop = f[0]

        best_model,best_model_name,best_scores,population,pop_scores,initial_scores,initial_best,scores_evolution,best_evolution = irace2(models, 
                                                                         X_train, 
                                                                         y_train, 
                                                                         stop, 
                                                                         stat_test, 
                                                                         parameters_dict, 
                                                                         scoring='f1',
                                                                         cv=split,
                                                                         show_gen=show_gen)

        best_model.fit(X_train,y_train)
        y_pred = best_model.predict(X_test)

        avg_scores = np.mean([np.mean(scores) for scores in list(pop_scores.values())])
        initial_scores = np.mean([np.mean(scores) for scores in list(initial_scores.values())])
        ini_best = np.mean(initial_best)
        final_best = np.mean(best_scores)
        initial = initial_scores
        final = avg_scores

        exp_list[exp_number]['data_id'] = int(data_id)
        exp_list[exp_number]['stat_test'] = stat_test.__name__
        exp_list[exp_number]['split'] = int(split)
        exp_list[exp_number]['iter_max'] = int(stop)
        exp_list[exp_number]['type_partition'] = 'cv'
        exp_list[exp_number]['best_model_name'] = type(best_model).__name__
        exp_list[exp_number]['test_score'] = f1_score(y_test,y_pred)
        exp_list[exp_number]['best_evolution'] = best_evolution
        exp_list[exp_number]['scores_evolution'] = scores_evolution 
        exp_number+=1

        row = [data_id,stat_test.__name__,split,stop,'cv',type(best_model).__name__,f1_score(y_test,y_pred),ini_best,final_best,initial,final]
        res.append(row)

        best_model,best_model_name,best_scores,population,pop_scores,initial_scores,initial_best,scores_evolution,best_evolution = irace2(models, 
                                                                         X_train, 
                                                                         y_train, 
                                                                         stop, 
                                                                         stat_test, 
                                                                         parameters_dict, 
                                                                         scoring='f1',
                                                                         r=split,
                                                                         show_gen=show_gen)
        
        
        best_model.fit(X_train,y_train)
        y_pred = best_model.predict(X_test)

        avg_scores = np.mean([np.mean(scores) for scores in list(pop_scores.values())])
        initial_scores = np.mean([np.mean(scores) for scores in list(initial_scores.values())])
        ini_best = np.mean(initial_best)
        final_best = np.mean(best_scores)
        initial = initial_scores
        final = avg_scores

        exp_list[exp_number]['data_id'] = int(data_id)
        exp_list[exp_number]['stat_test'] = stat_test.__name__
        exp_list[exp_number]['split'] = int(split)
        exp_list[exp_number]['iter_max'] = int(stop)
        exp_list[exp_number]['type_partition'] = 'tt'
        exp_list[exp_number]['best_model_name'] = type(best_model).__name__
        exp_list[exp_number]['test_score'] = f1_score(y_test,y_pred)
        exp_list[exp_number]['best_evolution'] = best_evolution
        exp_list[exp_number]['scores_evolution'] = scores_evolution 
        exp_number+=1

        row = [data_id,stat_test.__name__,split,stop,'tt',type(best_model).__name__,f1_score(y_test,y_pred),ini_best,final_best,initial,final]
        res.append(row)

        df = pd.DataFrame(res, columns = ['data_id','stat_test','n','max_iter','type','best_model','f1_score','ini_best','final_best','initial','final'])

        df.to_csv(csv_file)

        with open(json_file, "w") as final:
            json.dump(exp_list, final, cls=NumpyArrayEncoder)
