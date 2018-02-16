#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 19:46:40 2018

@author: Zander
"""
import pickle
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import imp

building_tools = imp.load_source('building_tools', '../../building_tools/building_tools.py')

# Load Data
with open('../transform_indiv_on_correlation/agg_df.pickle', 'rb') as file:
    df = pickle.load(file)

X = df.drop(labels=['poor'], axis=1)
Y = df['poor']

# Reserve 30% of Data for a Holdout Validation Set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=541030, stratify=Y)

# Prepare for parallelization
num_cores, m, n, bootstrap_m_1, bootstrap_m_2 = building_tools.split_work(x_train.shape[0], gridsearch_min_iter=16,
                                                                          scoring_min_iter=96)
print('Cores: {}, m: {}, n: {}, bootstrap_m_1: {}, bootstrap_m_2: {}'.format(num_cores, m, n, bootstrap_m_1,
                                                                             bootstrap_m_2))

# Set the scoring metric to use
gridsearch_scoring = 'neg_log_loss'

# Decision Tree Classifier
tree_params = {'criterion': ['gini', 'entropy'],
               'max_depth': [4, 5, 6, 7, 8],
               'min_samples_leaf': [3, 5, 7]}
tree_init = DecisionTreeClassifier()

# Gaussian Naive Bayes Classifier
gnb_params = {}
gnb_init = GaussianNB()

# K-Nearest Neighbors Classifier
knn_params = {'n_neighbors': [3, 5, 7, 9],
              'p': [1, 2],
              'n_jobs': [-1]}
knn_init = KNeighborsClassifier()

# Logistic Regression Classifier
log_params = {'C': [1, 10, 100, 1000],
              'tol': [1e-1, 1e-2, 1e-3, 1e-4],
              'penalty': ['l1', 'l2'],
              'class_weight': ['balanced'],
              'n_jobs': [-1]}
log_init = LogisticRegression()

# Random Forest Classifier
rfc_params = {'n_estimators': np.arange(50, 501, 50)}
rfc_init = RandomForestClassifier(n_jobs=-1, max_features='sqrt')

# Complete Random Trees Classifier
etc_params = {'n_estimators': np.arange(50, 501, 50)}
etc_init = ExtraTreesClassifier(n_jobs=-1, max_features='sqrt')

# AdaBoost Classifier with Decision Tree
base_tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=7)
adt_params = {'n_estimators': np.arange(50, 501, 50)}
adt_init = AdaBoostClassifier(base_estimator=base_tree)

# Multi-Layer Perceptron
mlp_params = {'hidden_layer_sizes': [(100,), (100,30)],
              'activation': ['tanh', 'relu'],
              'alpha': [0.0001, 0.001, 0.01, 0.1],
              'learning_rate_init': [0.0001, 0.001, 0.01, 0.1, 1.0],
              'learning_rate': ['adaptive'],}
mlp_init = MLPClassifier()

# Gradient Boosting
gbc_params = {'n_estimators': np.arange(50, 501, 50),
              'max_depth': [3, 4, 5, 6],}
gbc_init = GradientBoostingClassifier()

# Combine various parameters for each model into lists
model_names = ['tree', 'gnb', 'knn', 'log', 'rfc', 'etc', 'adt', 'mlp', 'gbc']
model_params = [tree_params, gnb_params, knn_params, log_params, rfc_params, etc_params, adt_params, mlp_params, gbc_params]
model_inits = [tree_init, gnb_init, knn_init, log_init, rfc_init, etc_init, adt_init, mlp_init, gbc_init]

# Perform GridSearchCV and CV Scoring
results = building_tools.bootstrap_tuning(x_train, y_train, model_names, model_params, model_inits,
                                          bootstrap_m_1=bootstrap_m_1, bootstrap_m_2=bootstrap_m_2,
                                          gridsearch_scoring=gridsearch_scoring, save=True)

# Pickle the data for future use
with open(os.path.join('results', 'full_results.pickle'), 'wb') as f:
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)




