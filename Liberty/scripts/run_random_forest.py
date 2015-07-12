from __future__ import print_function

import data_prep as dp
import numpy as np

from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


def run_random_forest(X_train, y_train, X_test, y_test):
    print("Fitting Random Forest")
    X_train = dp.factorize_variables(X_train)
    X_test = dp.factorize_variables(X_test)

    max_features = [None, 'auto', 'log2']
    params = {'criterion': ['gini'],
          'random_state': [1234],
          'n_estimators': [100, 200],
          'max_features': max_features,
          'oob_score': [False],
          # EOSL -- just fit max depth trees. not going to overfit
          'max_depth': [None, 10],
          'n_jobs': [1],
          }

    cv_func, y  = dp.get_kfold_obj(y_train, k = 3)
    grid = GridSearchCV(RandomForestClassifier(), params, cv=cv_func, 
        verbose=2)
    grid.fit(X_train, y_train.values)

    print(grid.best_score_)
    print(grid.best_estimator_)
    
    print("Training set score {}".format(grid.score(X_train,y_train)))
    print("Test set score {}".format(grid.score(X_test,y_test)))

    return grid




