#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 17:36:09 2018

@author: Zander
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV, cross_validate
from scipy.stats import ttest_ind, norm
from multiprocessing import cpu_count
import time
import pickle
import os

from sklearn.metrics import log_loss, accuracy_score, f1_score, precision_score, recall_score, make_scorer, precision_recall_fscore_support


class Bootstrap_Indices:
    '''A cross-validation iterator that will work with GridSearchCV.
    The iterator uses Sci-Kit Learn's resample method to sample with replacement and
    uses Numpy's in1d method to determine which indices will be in the test set. Also
    relies on Numpy's arange method.
    
    @ Parameters:
    ---------------
    n: number of observations
    m: number of times to sample with replacement
    
    @ Returns:
    ---------------
    an iterable of idx_train, idx_test indices
    '''
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.i = 0
    
    def __len__(self): 
        return self.m
        
    def __iter__(self):
        idx = np.arange(self.n)
        while self.i < self.m:
            self.i += 1
            idx_train = resample(idx)
            mask = np.in1d(idx, idx_train, invert=True)
            idx_test = idx[mask]
            yield idx_train, idx_test
            
def tune_parameters (x, y, clf, params, **kwargs):
    '''A simple wrapper for GridSearchCV to speed up repetitive process.
    
    @ Parameters:
    ---------------
    x: array of explanatory variables
    y: array of target variable
    clf: an initialized Sci-Kit Learn classifier
    params: dictionary of parameters over which to search
    
    @ **kwargs (optional):
    ----------------------
    folds: number of cross-validation folds to use (see GridSearchCV docs for other options)
    scoring_metric: a scorer to evaluate hyperparameters (if None, will use model default)
    n_jobs: number of parallel processing units to employ (defaults: -1)
    
    @ Returns:
    ----------------------
    best_params_: a dictionary of the best parameters
    cv_results_: a dictionary of all results
    '''
    
    # Gather data from kwargs, if supplied; otherwise, use defaults
    folds = kwargs.pop('folds', 10)
    scoring_metric = kwargs.pop('scoring_metric', None)
    n_jobs = kwargs.pop('n_jobs', -1)
    
    # Initialize grid search instance
    gs = GridSearchCV(clf, param_grid=params, cv=folds, scoring=scoring_metric, n_jobs=n_jobs, refit=False, verbose=2, return_train_score=True)
    # Fit grid search model
    gs.fit(x, y)
    
    return gs.best_params_, gs.cv_results_

def split_work (num_observations, **kwargs):
    '''A function to determine an appropriate number of bootstrap iterations
    to reduce unused CPU load (appropriate for AWS EC2 instances). For example,
    if the minimum number of bootstrap folds for GridSearchCV is set to 20, but
    there are 16 CPU cores available, the function will set the number of folds
    to 32, which meets the minimum but also makes use of all cores (i.e., we do
    not pay for 12 cores that go unused while 4 are busy).
    
    @ **kwargs (optional):
    -----------------------
    num_cores: (defaults to total number of system CPU cores)
    gridsearch_min_iter: minimum number of bootstrap iterations for GridSearchCV (default: 20)
    scoring_min_iter: minimum number of bootstrap iterations for scoring (default: 200)
    dampener: an integer value which to subtract from the total number of cores used
    
    '''
    # Get optional arguments
    dampener = kwargs.pop('dampener', 0)
    num_cores = kwargs.pop('num_cores', cpu_count()) - dampener
    gridsearch_min_iter = kwargs.pop('gridsearch_min_iter', 20)
    scoring_min_iter = kwargs.pop('scoring_min_iter', 200)
    threshold = kwargs.pop('threshold', 10000)
    
    # Determine core-friendly numbers for tasks
    m = np.ceil((num_observations / num_cores))
    batch_multiplier = np.ceil(m / threshold)
    m = int(m / batch_multiplier)
    n_estimators = int(np.floor(num_observations/m))
    bootstrap_m_1 = int(np.ceil(gridsearch_min_iter / num_cores)) * num_cores
    bootstrap_m_2 = int(np.ceil(scoring_min_iter / num_cores)) * num_cores
    
    return num_cores, m, n_estimators, bootstrap_m_1, bootstrap_m_2

def bootstrap_tuning (x, y, model_names, model_params, model_inits, **kwargs):
    '''docstring
    
    @ Parameters:
    -----------------
    x: matrix of observations and explanatory variables
    y: matrix of observations and target variables
    model_names: an ordered list of model names
    model_params: an ordered list of parameter dictionaries for use in GridSearchCV
    model_inits: an ordered list of initialized Sci-Kit Learn algorithms
    
    
    @ **kwargs:
    -----------------
    gridsearch_scoring: Scoring parameter for use in GridSearchCV
    cv_scoring: Scoring parameter(s) for use in cross validation
    bootstrap_m_1: number of times to sample with replacement during GridSearchCV (default: 20)
    bootstrap_m_2: number of times to sample with replacement during scoring (default: 200)
    '''
    # Search for optional arguments
    gridsearch_scoring = kwargs.pop('gridsearch_scoring', 'roc_auc')
    cv_scoring = kwargs.pop('cv_scoring', {'accuracy': 'accuracy',
                                          'f1_weighted': 'f1_weighted',
                                          'precision_0': make_scorer(precision_score, pos_label=0, average='binary'),
                                          'recall_0': make_scorer(recall_score, pos_label=0, average='binary'),
                                          'f1_0': make_scorer(f1_score, pos_label=0, average='binary'),
                                          'precision_1': make_scorer(precision_score, pos_label=1, average='binary'),
                                          'recall_1': make_scorer(recall_score, pos_label=1, average='binary'),
                                          'f1_1': make_scorer(f1_score, pos_label=1, average='binary'),
                                          'log_loss': make_scorer(log_loss, needs_proba=True),
                                          })
    bootstrap_m_1 = kwargs.pop('bootstrap_m_1', 20)
    bootstrap_m_2 = kwargs.pop('bootstrap_m_2', 200)
    num_cores = kwargs.pop('num_cores', -1)
    save = kwargs.pop('save', False)
    
    # Determine parameters for parallel processing
    #num_cores, bootstrap_m_1, bootstrap_m_2 = split_work()
    bootstrap_n = x.shape[0]
    
    # Initialize dictionary for storage
    models = {}
    # Iterate through models
    for clf, params, init in zip(model_names, model_params, model_inits):
        # Save data and parameters
        models[clf] = {}
        models[clf][clf + '_params'] = params
        models[clf][clf + '_init'] = init
        if params:
            # Initialize first bootstrap iterator
            bootstrap_1 = Bootstrap_Indices(bootstrap_n, bootstrap_m_1)
            # Progress
            t0 = time.time()
            # Tune hyperparameters
            models[clf]['best_params'], models[clf]['gs_results'] = tune_parameters(x, y, init, params, folds=bootstrap_1, scoring_metric=gridsearch_scoring, n_jobs=num_cores)
            # Progress
            models[clf]['gs_time'] = time.time() - t0
            # Update each of the model parameters with best values
            models[clf]['model'] = models[clf][clf + '_init']
            for key, value in models[clf]['best_params'].items():
                models[clf]['model'].__setattr__(key, value)
        else:
            models[clf]['model'] = models[clf][clf + '_init']
        t0 = time.time()
        # Initialize second bootstrap iterator
        bootstrap_2 = Bootstrap_Indices(bootstrap_n, bootstrap_m_2)
        # Gather an array of scores for significance testing using bootstrapping
        models[clf]['cv_scores'] = cross_validate(models[clf]['model'], x, y, scoring=cv_scoring, cv=bootstrap_2, return_train_score=False, n_jobs=num_cores)
        # Notify progress
        models[clf]['cv_time'] = time.time() - t0
        # Optional save model
        if save:
            with open(os.path.join('results', clf + '.pickle'), 'wb') as f:
                pickle.dump(models[clf], f, pickle.HIGHEST_PROTOCOL)
    return models

def comparison (*args, **kwargs):
    '''This function prints a comparison of scores.
    
    @ Parameters:
    --------------
    *args: Any number of arrays for which you want to compare confidence intervals
    **kwargs:
        C: Confidence Interval (default = 0.95)
        labels: ordered list of labels to apply to ordered args
        scorer: name of the scoring function being compared.'''
    C = kwargs.pop('C', 0.95)
    labels = kwargs.pop('labels', ['Model {}'.format(x) for x in range(len(args))])
    scorer = kwargs.pop('scorer', None)
    max_l = len(max(labels, key=len))
    
    print('Comparison of {} Scores'.format(scorer))
    for arg, label in zip(args, labels):
        # Calculate confidence interval
        conf = norm.interval(C, loc=arg.mean(), scale=arg.std())        
        print('  {0:{1}}  Mean: {2:.2f}  Std: {3:.2f}  95%Conf: ({4:.2f}, {5:.2f})'.format(label, max_l, arg.mean(), arg.std(), conf[0], conf[1]))
        
def plot_distributions(*args, **kwargs):
    '''This function plots the distribution and confidence interval of all arrays passed to it.
    
    @ Parameters:
    --------------
    *args: Any number of arrays for which you want to visualize the distribution
    **kwargs:
        C: Confidence Interval (default = 0.95)
        num_bins: number of bins to use in histogram (default=20)
        labels: ordered list of labels to apply to ordered args
        colors: ordered list of colors to apply to ordered args
        x_lim: list representing the left and right x limits
        plot_width: an integer value representing width in inches
        plot_height: an integer value representing height in inches
        limit_outliers: if True, histograms will ignore any outliers (1.5 * IQR). Default=False.
        alpha: alpha value for plot opacity.
        scorer: name of the scoring function being compared.
        conf_lines: if 'True' will plot vertical lines to show confidence intervals
    '''

    # Gather data from kwargs, if supplied; otherwise, use defaults
    C = kwargs.pop('C', 0.95)
    num_bins = kwargs.pop('num_bins', 20)
    labels = kwargs.pop('labels', ['Model {}'.format(x) for x in range(len(args))])
    colors = kwargs.pop('colors', cm.rainbow(np.linspace(0,1,len(args))))
    x_lim = kwargs.pop('x_lim', [None, None])
    plot_width = kwargs.pop('plot_width', 14)
    plot_height = kwargs.pop('plot_height', 5)
    limit_outliers = kwargs.pop('limit_outliers', False)
    alpha = kwargs.pop('alpha', 0.6)
    scorer = kwargs.pop('scorer', None)
    conf_lines = kwargs.pop('conf_lines', False)
    
    # Set plot dimensions
    plt.rcParams['figure.figsize'] = (plot_width, plot_height)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.20, .80)})

    # Plot boxplots
    ax1.boxplot(args, vert=False, sym='rs', labels=labels)
    # Remove y-axis and set overall title (looks better than a sup title)
    #ax1.axes.get_yaxis().set_visible(False)
    if scorer:
        ax1.set_title('Distribution of {} Scores'.format(scorer))
    else:
        ax1.set_title('Distribution of Scores')
    
    # Plot histograms
    for arg, color, label in zip(args, colors, labels):
        # Determine whisker_range for histograms
        if limit_outliers == True:
            p_25 = np.percentile(arg, 25)
            p_75 = np.percentile(arg, 75)
            iqr = p_75 - p_25
            whisker_range = (p_25 - (iqr * 1.5), p_75 + (iqr * 1.5))
        else:
            whisker_range = None
        # Calculate confidence interval
        conf = norm.interval(C, loc=arg.mean(), scale=arg.std())
        # Prepare label
        l = '({0: .2f}, {1:.2f}) - {2}'.format(conf[0], conf[1], label)
        # Plot histogram
        N, bins, patches = ax2.hist(arg, bins=num_bins, range=whisker_range, color=color, edgecolor='black', alpha=alpha, label=l)
        # Plot confidence intervals
        if conf_lines == True:
            for val in conf:
                ax2.axvline(x=val, color=color, linestyle='--', alpha=alpha)

    # Set plot attributes
    plt.legend(loc='best')
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.xlim(x_lim)
    plt.show()
    
def plot_decision_boundaries (X, Y, x_lab, y_lab, *args, **kwargs):
    '''Plots the decision boundaries of any number of classifiers.
    The following code is originally based on the visualization technique
    presented by Sci-Kit Learn, at 
    http://scikit-learn.org/stable/auto_examples/ensemble/plot_voting_decision_regions.html.
    Significant modifications have been made.
    
    @ Parameters:
    ---------------
    X: explanatory variables
    Y: target feature
    x_lab: the explanatory variable to plot on the x-axis
    y_lab: the explanatory variable to plot on the y-axis
    *args: list of initialized classifiers
    
    @ **kwargs (optional):
    ---------------
    offset: +/- amount to extend mesh grid
    labels: list of classifier names
    num_cols: number of columns in plot
    ax_width: width of each column
    ax_height: height of each column
    '''
    # Pull kwargs, if offered
    offset = kwargs.pop('offset', 0.05)
    labels = kwargs.pop('labels', ['Model {}'.format(x) for x in range(len(args))])
    num_cols = kwargs.pop('num_cols', 2)
    ax_width = kwargs.pop('ax_width', 7)
    ax_height = kwargs.pop('ax_height', 5)
    
    # Determine basic sizes
    num_items = len(args)
    num_rows = int(np.ceil(num_items / num_cols))
    
    # Set plot size
    plot_width = num_cols * ax_width
    plot_height = num_rows * ax_height
    plt.rcParams['figure.figsize'] = (plot_width, plot_height)
    
    # Ensure Target Classification is an array
    cls = Y.as_matrix()
    # Gather data
    cols = [x_lab, y_lab]
    x = X[x_lab]
    y = X[y_lab]
    
    # Set up mesh grid for contour predictions
    x_min, x_max = x.min() - offset, x.max() + offset
    y_min, y_max = y.min() - offset, y.max() + offset
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Prepare subplots in grid
    f, axarr = plt.subplots(num_rows, num_cols, sharex='col', sharey='row')
    
    item_locator = 0
    # Iterate through rows and columns
    for r in range(num_rows):
        for c in range(num_cols):
            # Fit algorithm
            args[item_locator].fit(X[cols], Y)
            # Make predictions across mesh grid space
            z = args[item_locator].predict(np.c_[xx.ravel(), yy.ravel()])
            z = z.reshape(xx.shape)
            # Plot contour and scatter
            axarr[r, c].contourf(xx, yy, z, alpha=0.4)
            axarr[r, c].scatter(x, y, c=cls)
            axarr[r, c].set_title(labels[item_locator])
            item_locator += 1

    plt.show()
    
def compare_scores (*args, alternate='less', **kwargs):
    '''Function performs a series of t-tests, comparing the first element in *args to all 
    other elements. To avoid increasing the likelihood of making a Type I error, we use the
    Bonferroni method, which divides the alpha value by the number of hypotheses tested.
    
    The T-test function provided in SciPy always produces a two-side test, but the user
    can choose a one-sided test by supplying the necessary parameter.
    
    We would reject the null hypothesis when:
        * H0: a <= b, Ha: a > b : reject H0 when p/2 < alpha and t > 0
        * H0: a >= b, Ha: a < b : reject H0 when p/2 < alpha and t < 0
        * H0: a = b, Ha: a != b : reject H0 when p < alpha
        
    @ Parameters:
    ---------------
    *args: list of score arrays, where first in list is compared to all others
    alternate: value of "less", "more", or "unequal" - will determine test that is run.
        User should assume "less" means that they are testing if the scores of the first
        argument are less than the scores of all other arguments.
    
    @ **kwargs (optional):
    ---------------
    alpha: alpha value for testing (function will automatically update if required for
        one-sided test and to take into account Bonferroni method)
    labels: a list of names of the score arrays
    '''
    # Gather keyword arguments, if any
    alpha = kwargs.pop('alpha', 0.05)
    labels = kwargs.pop('labels', ['Model {}'.format(x) for x in range(len(args))])
    
    # Determine appropriate signs for hypotheses
    if alternate == 'less':
        hyp_sign_1 = '>='
        hyp_sign_2 = '<'
        q = 2
    elif alternate == 'more':
        hyp_sign_1 = '<='
        hyp_sign_2 = '>'
        q = 2
    elif alternate == 'unequal':
        hyp_sign_1 = '='
        hyp_sign_2 = '!='
        q = 1
    
    # Determine the Bonferroni correction based on number of hypotheses to test
    m = len(args) - 1
    bonferroni = alpha / m
    
    for arg, label in zip(args[1:], labels[1:]):
        # Calculate t and p
        t_statistic, p_value = ttest_ind(args[0], arg)
        hyp_state = 'Ho: {0} {1} {2}\nHa: {0} {3} {2}\nReject Ho when p/2 < {4} and t {3} 0'.format(labels[0], hyp_sign_1, label, hyp_sign_2, bonferroni)
        hyp_test = '\tT-statistic: {0:.2f}, P-value: {1}'.format(t_statistic, p_value)
        # Determine whether to reject null hypothesis
        if p_value / q < bonferroni:
            if alternate == 'less' and t_statistic < 0:
                res = 'Reject'
            elif alternate == 'more' and t_statistic > 0:
                res = 'Reject'
            elif alternate == 'unequal':
                res = 'Reject'
            else:
                res = 'Fail to reject'
        else:
            res = 'Fail to reject'
        hyp_result = '\t{0} the null hypothesis.'.format(res)
        print('\n'.join([hyp_state, hyp_test, hyp_result]) + '\n')