import pandas as pd
import numpy as np
import csv
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import hp
import hyperopt.pyll.stochastic as hps
import time



# Load and inspect data
pathname = "C:/Users/mat/Documents/"
df = pd.read_csv(pathname + 'boostmatch_2.csv')
# Get dummies
#df = pd.get_dummies(df, columns=['county_class'])
#df = pd.get_dummies(df, columns=['state'])
# Drop missings / unwanted columns
#df = df.dropna() 
#df = df.drop('riskobj', 1)
#df = df.drop('mitigation', 1)
#df = df.drop('inscont', 1)
df = df.drop('Unnamed: 0', 1)
df = df.drop('nocon1', 1)
# Train/test split
np.random.seed(7)
# Declare y,X
y = df['nocon']
X = df.drop('nocon', 1)
print(X.head(5))
print("check1")

#########################################
# Split test/train
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20, random_state=7)
print("check2")
#########################################################
# LightGBM
# https://lightgbm.readthedocs.io/en/latest/
import lightgbm as lgb
import random
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
print("check3")
# create dataset for lightgbm
lgb_train = lgb.Dataset(xtrain, ytrain)
lgb_eval = lgb.Dataset(xtest, ytest, reference=lgb_train)
print("check4")

import lightgbm as lgb
from hyperopt import STATUS_OK

N_FOLDS = 5

# Create the dataset
train_set = lgb.Dataset(xtrain, ytrain)

out_file = 'gbm_trials.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)
# Write the headers to the file
writer.writerow(['loss', 'params', 'estimators', 'train_time'])
of_connection.close()

def objective(params, n_folds = N_FOLDS):
    """Objective function for Gradient Boosting Machine Hyperparameter Tuning"""

    start = time.time()
    
    # Perform n_fold cross validation with hyperparameters
    # Use early stopping and evalute based on ROC AUC
    cv_results = lgb.cv(params, train_set, nfold = n_folds, num_boost_round = 10000, 
                        early_stopping_rounds = 100, metrics = 'auc', seed = 50)
    run_time = time.time() - start
    # Extract the best score
    best_score = max(cv_results['auc-mean'])
    
    # Loss must be minimized
    loss = 1 - best_score
    n_estimators = int(np.argmax(cv_results['auc-mean']) + 1)
    
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, params,  n_estimators, run_time])

    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'status': STATUS_OK}
    

space = {
    'objective':'binary',
    'verbose' : -1,
    'boosting_type' : 'gbdt',
    'min_data_in_leaf': 1, 
    'min_sum_hessian_in_leaf':0,
    'learning_rate' : hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
    'max_depth' : hp.choice('max_depth', np.arange(1, 5, dtype=int)),
    'lambda_l1': hp.quniform('lambda_l1',0.7, 1,0.1),
    'num_leaves': hp.choice('num_leaves',np.arange(30, 51,1, dtype=int)),
    'feature_fraction': hp.uniform('feature_fraction',0.1, 0.99),
    'bagging_fraction': hp.uniform('bagging_fraction',0.3, 0.9), 
    'max_bin': hp.choice('max_bin',np.arange(2, 10, dtype=int)),
    'bagging_freq': hp.choice('bagging_freq',np.arange(2, 50, dtype=int)),
}
example = hps.sample(space)
tpe_algorithm = tpe.suggest
bayes_trials = Trials()



# Write to the csv file ('a' means append)


from hyperopt import fmin

MAX_EVALS = 500

# Optimize
best = fmin(fn = objective, space = space, algo = tpe.suggest, 
            max_evals = MAX_EVALS, trials = bayes_trials)

# File to save first results
bayes_trials_results = sorted(bayes_trials.results, key = lambda x: x['loss'])
print(bayes_trials_results[:2])