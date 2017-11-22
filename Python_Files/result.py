# -*- coding: utf-8 -*-

import numpy as np
import xgboost as xgb

#x_train = np.load('../Extracted_Features/lex.npy')

#x_train = np.load('../Extracted_Features/read.npy')

x_train = np.load('../Extracted_Features/numeric.npy')

y_train = np.load('../Extracted_Features/label.npy')

# Set our parameters for xgboost
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 4

d_train = xgb.DMatrix(x_train, label=y_train)
#d_valid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(d_train, 'train')]

bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)