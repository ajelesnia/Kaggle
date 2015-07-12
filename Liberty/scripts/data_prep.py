from __future__ import print_function

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn import preprocessing, metrics, cross_validation
from sklearn.cross_validation import train_test_split, StratifiedKFold



def read_data(file_path):
    return pd.read_csv(file_path)

def get_target_variable(data, dep_var_name):
    assert str(dep_var_name), "Target name should be string"
    target = data.pop(dep_var_name)
    return target, data

def mean_imputation(data,missing_values):
    return data.Imputer(missing_values=missing_values, strategy='mean', axis=0, verbose=0, copy=True)

def mode_imputation(data,missing_values):
    return data.Imputer(missing_values=missing_values, strategy='mean', axis=0, verbose=0, copy=True)

def median_imputation(data,missing_values):
    return data.Imputer(missing_values=missing_values, strategy='mean', axis=0, verbose=0, copy=True)

def split_train_test(data, train_size = 0.80):
    return train_test_split(data, train_size=train_size, random_state = 6279)


def factorize_variables(data):
    #works but has some issues in ipython notebook
    column_names = data.columns
    for column in column_names:
        try:
            int(data[column].ix[1])
        except:
            a, b = pd.factorize(data[column])
            data = data.drop(column, axis=1)
            data[column] = a
    return data


def get_kfold_obj(target, k = 5):
    """
    Returns a list of K-Fold objects
    """
    y = target
    cv_funcs = StratifiedKFold(y, n_folds=k, random_state=6279)

    return cv_funcs, y


def print_columns_with_missing(data):
    for i in data.columns:
        if True in set(pd.isnull(data[i])):
            print(i) 


def pickle_model(obj, fname):
    joblib.dump(obj, fname, compress=3)


def unpickle_model(fname):
    return joblib.load(fname)

