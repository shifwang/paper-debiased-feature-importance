#!/usr/bin/env python
# coding: utf-8

import shap
import pandas as pd
import scipy as sp
import numpy as np
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer, MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, mean_squared_error
import matplotlib.pyplot as plt
from treeinterpreter.feature_importance import feature_importance as FI
from treeinterpreter.feature_importance import MDA

def sigmoid(x):
    x *= 2
    return np.exp(x) / (np.exp(x) + np.exp(-x))

def permute(X, noisy_features):
    tmp = X.copy()
    for j in range(X.shape[1]):
        if noisy_features[j] == 1:
            tmp[:, j] = np.random.permutation(tmp[:, j])
    return tmp

def f(X, noisy_features):
    probs = sigmoid(np.mean(X[:, noisy_features == 0], 1))
    return np.array([np.random.choice([0, 1], 1, p=[1 - prob, prob]) for prob in probs]).flatten()

# # Compare different methods in terms of feature selection using simulated data
debiased_list, gini_list, shap_list, mda_list = [], [], [], []

for i in range(40):
    # ### load data
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
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = permute(X_train, noisy_features)
    X_test = permute(X_test, noisy_features)
    y_train = f(X_train, noisy_features)
    y_test = f(X_test, noisy_features)
    
    rf = rfc(n_estimators=100, max_features = 8, min_samples_leaf=100)
    rf.fit(X_train, y_train)
    gini_imp = rf.feature_importances_

    y_train_one_hot = OneHotEncoder().fit_transform(y_train[:, np.newaxis]).todense()
    debiased_fi_mean, debiased_fi_std = FI(rf, X_train, y_train_one_hot, type='oob', normalized=False)
    debiased_imp = debiased_fi_mean

    MDA_imp, _ = MDA(rf, X_test, y_test[:, np.newaxis], type='test', n_trials=10)

    explainer = shap.TreeExplainer(rf)
    samples = np.random.choice(range(X_train.shape[0]), 100)
    shap_values = explainer.shap_values(X_train[samples, :])
    shap.summary_plot(shap_values, X_train[samples, :], plot_type="bar")

    shap_imp = np.mean(abs(shap_values[0]), 0) + np.mean(abs(shap_values[1]), 0)

    debiased_imp[debiased_imp < 0] = 0
    MDA_imp[MDA_imp < 0] = 0

    debiased_list.append(roc_auc_score(noisy_features, - debiased_imp))
    gini_list.append(roc_auc_score(noisy_features, - gini_imp))
    shap_list.append(roc_auc_score(noisy_features, - shap_imp))
    mda_list.append(roc_auc_score(noisy_features, - MDA_imp))

np.savez('../intermediate/02_simulation_results_correlated_shallow.npz', debiased_list = debiased_list, gini_list=gini_list, shap_list=shap_list, mda_list=mda_list)
