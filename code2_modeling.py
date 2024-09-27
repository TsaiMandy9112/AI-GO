#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 14:51:50 2023

@author: 
"""

import pandas as pd
from sklearn.model_selection import train_test_split

#--- Generative Ensemble Learning(集成式學習) Algorithm for House Pirce Prediciton 
# Bootstrap aggregation (bagging)
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor 
# Gradient bootsting (梯度提升)
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb

# Metric
from sklearn.metrics import r2_score                         #內部測試使用
from sklearn.metrics import mean_absolute_percentage_error   #比賽使用

#------------------------------------------------------------------------
# load dataset (clean)
#------------------------------------------------------------------------
df=pd.read_csv('clean_dataset.csv')


#------------------------------------------------------------------------
# Data Split: training & test data
#------------------------------------------------------------------------
X=df.drop(['unit_price'],axis=1) # features
y=df['unit_price']               # labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=2023)


#------------------------------------------------------------------------
# Modeling & Performace Evaluation
#------------------------------------------------------------------------
# Create a list of regressors
regressors = [
    ('Bagging Regressor', BaggingRegressor(random_state=2023)),
    ('ExtraTrees Regressor', ExtraTreesRegressor(random_state=2023)),
    ('Random Forest', RandomForestRegressor(random_state=2023)),
    ('Gradient Boosting Regressor', GradientBoostingRegressor(random_state=2023)),
    ('XGB Regressor', xgb.XGBRegressor(random_state=2023)),
    ('LightGBM Regressor', lgb.LGBMRegressor(objective='regression', metric='l1', seed=2023))   
]

#modeling
for name, reg in regressors:
    reg.fit(X_train, y_train)     # Train the regressor
    y_pred=reg.predict(X_test)    # Predict on the test set
    # evaluate the performace
    r2=r2_score(y_test, y_pred)  
    mape=mean_absolute_percentage_error(y_test, y_pred)    
    print('{0}:\n\tR2 Score(testing)={1:.4f}, MAPE(testing)={2:.4f}'.format(name, r2, mape))





