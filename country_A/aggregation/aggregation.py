#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

class Aggregate():	
    from scipy.stats import spearmanr, mode
    from sklearn.feature_selection import chi2
    from sklearn.ensemble import RandomForestClassifier
    
    def __init__ (self, X, **kwargs):
        self.X = X
        self.cat_cols = np.array([x for x in self.X.columns if '_cat_' in x])
        self.num_cols = np.array([x for x in self.X.columns if '_num_' in x])
        self.Y = kwargs.pop('Y', None)
        self.groupby = kwargs.pop('groupby', 'id')
        
    # Possible aggregation functions
    def any_of_indiv (df, groupby='id'):
        return df.groupby('id').any()

    def percent_of_indiv (df, groupby='id'):
        return df.groupby('id').sum() / df.groupby('id').count()

    def mode_of_indiv (df, groupby='id'):
        return df.groupby('id').agg(lambda x: Aggregate.mode(x)[0][0])
        #return df.groupby('id').agg(lambda x: scipy.stats.mode(x)[0][0])

    def mean_of_indiv (df, groupby='id'):
        return df.groupby('id').mean()

    def median_of_indiv (df, groupby='id'):
        return df.groupby('id').median()

    def max_of_indiv (df, groupby='id'):
        return df.groupby('id').max()
    
    def min_of_indiv (df, groupby='id'):
        return df.groupby('id').min()
    
    # Possible evaluation functions
    def chi_squared (x, y):
        return pd.Series(Aggregate.chi2(x, y)[0])
    
    def spearman (x, y):
        result = x.apply(Aggregate.spearmanr, axis=0, args=(y,))
        #result = x.apply(scipy.stats.spearmanr, axis=0, args=(y,))
        result = result.apply(lambda x: x[0])
        return result.reset_index(drop=True)
    
    def random_forest (x, y, **kwargs):
        n_estimators = kwargs.pop('rf_n_estimators', 10000)
        forest = Aggregate.RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)
        forest.fit(x, y)
        return forest.feature_importances_
    
    # Helper functions for comparisons
    def return_max (row, cols_to_test):
        '''Pandas apply function to return max function in row.'''
        temp = np.array([row[x] for x in cols_to_test])
        # If all values are NaN, doesn't matter which function. Return first.
        if np.all(np.isnan(temp)):
            return cols_to_test[0].split('_')[1]
        # nanargmax ignores NaNs
        return cols_to_test[np.nanargmax(temp)].split('_')[1]
    
    def compare (self, cols, funcs, eval_func, **kwargs):
        # If there is only one function, return it (no testing necessary)
        if len(funcs) == 1:
            return np.array([funcs[0] for x in cols])
        # A dataframe to be used for comparison purposes
        compare_df = pd.DataFrame(data={'Name': cols})
        for func_name in funcs:
            func = getattr(Aggregate, func_name + '_of_indiv')
            # Apply aggregation
            temp_df = func(self.X[cols], groupby=self.groupby)
            # Merge with Y to ensure indices match
            temp_df = pd.merge(temp_df, self.Y, left_index=True, right_index=True)
            # Gather transformed data
            X_transform = temp_df[cols]
            Y_transform = temp_df['poor']
            # Names
            rho = 'rho_' + func_name
            # Apply correlation
            compare_df[rho] = eval_func(X_transform, Y_transform, **kwargs)
        # List of columns to test
        cols_to_test = [x for x in compare_df.columns if 'rho' in x]
        # Return the best aggregate function for each categorical feature
        return np.array(compare_df.apply(Aggregate.return_max, axis=1, args=(cols_to_test,)))
    
    # Fit function
    def fit (self, **kwargs):        
        '''The fit function attempts all supplied aggregate functions for each column
        of type numeric or categorical. The transformed columns are then tested for
        suitability using the evaluation function (e.g. correlation with target variable).
        Finally, the top aggregate function for each column is stored as a dictionary
        lookup in self.col_to_func. 
        
        *Note, the 'mode' aggregation function is very slow, and has to make assumptions
            when there is not one clear mode in series.
                    
        Possible evaluation functions:
            'chi_squared', 'spearman', 'random_forest'
        
        Possible aggregation functions:
            'mean', 'median', 'mode', 'max', 'min', 'any', 'percent'
        '''
        # Gather and store options as class variables
        self.num_eval = getattr(Aggregate, kwargs.pop('num_eval', 'spearman'))
        self.cat_eval = getattr(Aggregate, kwargs.pop('cat_eval', 'chi_squared'))
        self.num_agg_funcs = kwargs.pop('num_agg_funcs', ['mean', 'median', 'max', 'min'])
        self.cat_agg_funcs = kwargs.pop('cat_agg_funcs', ['any', 'percent', 'mode'])
        self.rf_n_estimators = kwargs.pop('rf_n_estimators', 10000)
        # Gather column names
        self.cat_cols = np.array([x for x in self.X.columns if '_cat_' in x])
        self.num_cols = np.array([x for x in self.X.columns if '_num_' in x])
        # Determine best agg function
        self.cat_funcs = self.compare(self.cat_cols, self.cat_agg_funcs, self.cat_eval, **kwargs)
        self.num_funcs = self.compare(self.num_cols, self.num_agg_funcs, self.num_eval, **kwargs)
        # Combine transformations into dictionary mapping
        t1 = np.append(self.cat_cols, self.num_cols)
        t2 = np.append(self.cat_funcs, self.num_funcs)
        self.col_to_func = {key: value for key, value in zip(t1, t2)}
    
    # Apply transform
    def transform (self, **kwargs):
        '''The transform function applies aggregate functions to each column and returns
        a transformed dataframe. If no arguments are passed, this will transform the df
        originally passed when initializing the class. User can also pass a new df and
        a col_to_func dictionary mapping, useful when user wants to transform a different
        df using the same transformations as determined by the fit function.'''
        # Allow for ability to pass new X and previously defined functions
        X = kwargs.pop('X', self.X)
        col_to_func = kwargs.pop('col_to_func', self.col_to_func)
        # Transform and return
        return pd.concat([getattr(Aggregate, func + '_of_indiv')(X[col]) for col, func in col_to_func.items()], axis=1)
        
    def fit_transform (self, **kwargs):
        getattr(Aggregate, 'fit')(self, **kwargs)
        return getattr(Aggregate, 'transform')(self, **kwargs)