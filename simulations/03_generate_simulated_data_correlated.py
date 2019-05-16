import shap
import pandas as pd
import scipy as sp
import numpy as np
from irf.ensemble import wrf
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import matplotlib.pyplot as plt
from treeinterpreter.feature_importance import feature_importance as FI
from treeinterpreter.feature_importance import MDA
def sigmoid(x):
    x *= 1
    return np.exp(x) / (np.exp(x) + np.exp(-x))
def permute(X, noisy_features):
    tmp = X.copy()
    for j in range(X.shape[1]):
        if noisy_features[j] == 0:
            continue
        tmp[:, j] = np.random.permutation(sp.stats.rankdata(tmp[:, j]))
        maximum = max(tmp[:, j])
        tmp[:, j] = tmp[:, j] / maximum - .5
    return tmp
def rankdata(X):
    tmp = X.copy()
    for j in range(X.shape[1]):
        #tmp[:, j] = np.random.permutation(sp.stats.rankdata(tmp[:, j]))
        maximum = max(tmp[:, j])
        tmp[:, j] = tmp[:, j] / maximum - .5
    return tmp
def f(X, noisy_features):
    probs = sigmoid(np.mean(X[:, noisy_features == 0], 1) )
    return np.array([np.random.choice([0, 1], 1, p=[1 - prob, prob]) for prob in probs]).flatten()

for ind in range(40):
    X_train = np.loadtxt('../intermediate/02_enhancer/X_train.csv', delimiter=',')
    y_train = np.loadtxt('../intermediate/02_enhancer/y_train.csv', delimiter=',')
    X_test = np.loadtxt('../intermediate/02_enhancer/X_test.csv', delimiter=',')
    y_test = np.loadtxt('../intermediate/02_enhancer/y_test.csv', delimiter=',')
    n, m = X_train.shape
    names = np.arange(m)

    n_features = X_train.shape[1]
    n, m = X_train.shape
    names = np.arange(m)
    noisy_features = np.ones((n_features, ), dtype=int)
    noisy_features[np.random.choice(range(n_features), 5, replace=False)] = 0
    y_train = f(X_train, noisy_features)
    y_test = f(X_test, noisy_features)
    X_train = permute(X_train, noisy_features)
    X_test = permute(X_test, noisy_features)

    np.savetxt('../intermediate/02_enhancer/permuted{}_X_train_correlated.csv'.format(ind), X_train, delimiter=',')
    np.savetxt('../intermediate/02_enhancer/permuted{}_y_train_correlated.csv'.format(ind), y_train, delimiter=',')
    #np.savetxt('../intermediate/02_enhancer/permuted{}_X_test.csv'.format(ind), X_test, delimiter=',')
    #np.savetxt('../intermediate/02_enhancer/permuted{}_y_test.csv'.format(ind), y_test, delimiter=',')
    np.savetxt('../intermediate/02_enhancer/permuted{}_noisy_features_correlated.csv'.format(ind), noisy_features, delimiter=',')
    
