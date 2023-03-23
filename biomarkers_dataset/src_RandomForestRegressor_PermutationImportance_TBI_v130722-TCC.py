# Random Forest Regression Feature Importance

import pandas as pd

from pandas import read_csv
from numpy import set_printoptions
import matplotlib.pyplot as plt
import numpy as np
import csv

# random forest for feature importance on a regression problem
from sklearn.ensemble import RandomForestRegressor

#-----Banco-Carlisa----
filename = 'BancoTCE_Biomarkers_CARLISA_170522_TBI_v4_sem_header.csv.csv'
names = ['sex','age','Copeptin','Angiotensin 1-7','Angiotensin 2','TFG alpha','IFN alpha 2','GRO-KC','MCP-3','MDC','sCD40L','IL-4','MCP-1','BDNF','Catepsin D','sICAM-1','MPO','PDGF AA','NCAM','PDGF-AB BB','PAI-1','MMP-9','Fibroblast','Neuropilin-1','Lipocalin-2','VEGF','MIF','LIGHT','RAGE','Enolase NSE','NRG-1 beta 1','HADS anxiety','HADS depression']
#------------------------------------------------

dataframe = read_csv(filename, names=names)
array = dataframe.values

X = array[:,0:31]

variable_prediction = 31  # quando a variavel alvo: HADS anxiety
title_figure = "HADS anxiety"

#variable_prediction = 32  # quando a variavel alvo: HADS depression
#title_figure = "HADS depression"

simulations = 1
names2 = names[0:31]

y = array[:,variable_prediction]
print('variable_prediction')
print(y)

from sklearn.inspection import permutation_importance
 
v_import = np.zeros(X.shape[1])
for ite in range(0,simulations):
    # define the model
    print(f'-----ite =  {ite}-------')
    model = RandomForestRegressor(random_state = ite) #n_estimators = 1000
    # fit the model
    model.fit(X, y)
    # get importance
    importance = model.feature_importances_
    print(f'R-Squared = {model.score(X,y):.2f}')
    
    result = permutation_importance(model, X, y, n_repeats=100, random_state=ite) # 
    #print(f'result.importances_mean = {result.importances_mean}')
    #print(f'result.importances_std = {result.importances_std}')
    
    for ii in range(len(result.importances_mean)):
    #    v_import[ii] = v_import[ii] + (importance[ii]/simulations)
        v_import[ii] = v_import[ii] + (result.importances_mean[ii]/simulations)

for ii in range(len(v_import)):
    importance[ii] = v_import[ii]
   
#------------ summarize feature importance-----------------------
soma = 0
for ii in range(len(importance)):
    soma = soma + importance[ii]
        
num = len(importance) #importance
media_score = soma/num
print('\n')
print(f' media_score_total = {media_score:.3f}')

print('\n score >= media_score')
for ii in range(len(importance)):
    v = importance[ii]
    if (v) >= media_score : #media_score
        print(f' Feature {ii}: {names2[ii]}, Score: {v:.3f} ')

print('\n score')
for ii in range(len(importance)):
    v = importance[ii]
    print(f' Feature {ii}: {names2[ii]}, Score: {v:.3f} ')
    
ax = plt.subplot()
names2 = names[0:31]
pos = np.arange(len(names2))
ax.barh(pos,np.abs(importance), color='blue',edgecolor='black')
plt.yticks(pos, names2)
ax.set_title(title_figure, y=1.0, pad=-14)
plt.xlabel("mean score")
plt.show()

X = pd.DataFrame(X)
y = pd.DataFrame(y)

X.to_csv('X.csv',index=False)
y.to_csv('y.csv',index=False)

import pickle
with open('model.pck', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(model, f)

