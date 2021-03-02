import pandas as pd
import numpy as np
import csv



# Load and inspect data
pathname = "C:/Users/mat/Documents/"
df = pd.read_csv(pathname + 'boostmatch_2.csv')
print(df.head(5))
# Get dummies
#df = pd.get_dummies(df, columns=['county_class'])
#df = pd.get_dummies(df, columns=['state'])
# Drop missings / unwanted columns
#df = df.dropna() 
#df = df.drop('riskobj', 1)
#df = df.drop('mitigation', 1)
#df = df.drop('inscont', 1)
df = df.drop('Unnamed: 0', 1)
df = df.drop(['nocon','noc1','noc'], 1)
# Train/test split
np.random.seed(7)
# Declare y,X
y = df['nocon1']
X = df.drop('nocon1',1)
# X = df.drop('nocon', 1)
# X = df.drop('noc1', 1)
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
from sklearn.metrics import roc_curve
from sklearn.metrics import mean_squared_error
print("check3")
# create dataset for lightgbm
lgb_train = lgb.Dataset(xtrain, ytrain)
lgb_eval = lgb.Dataset(xtest, ytest, reference=lgb_train)
print("check4")
# # Perform random search for best hyperparameter
# best_auc = 0
# best_params = []
# # https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
# for i in range(0,50000):
#     try:
#         params = {
#                 'objective' :'binary',
#                 'learning_rate' : random.uniform(0.6, 0.9), #0.763
#                 'max_depth' : random.randint(1,5), # limit the max depth for tree model. This is used to deal with over-fitting when #data is small
#                 'lambda_l1': random.uniform(0.8, 1), # L1 regularization (>=0)
#                 'num_leaves' : random.randint(30,40), # max number of leaves in one tree (default=31)
#                 'feature_fraction': random.uniform(0.2, 0.5), # LightGBM will randomly select part of features on each iteration if feature_fraction smaller than 1.0. For example, if you set it to 0.8, LightGBM will select 80% of features before training each tree
#                 'bagging_fraction': random.uniform(0.2, 0.5), # like feature_fraction, but this will randomly select part of data without resampling
#                 'max_bin': random.randint(1,10), # max number of bins that feature values will be bucketed in small number of bins may reduce training accuracy but may increase general power (deal with over-fitting)
#                 'bagging_freq': random.randint(30,40), # frequency for bagging: 0 means disable bagging; k means perform bagging at every k iteration
#                 'min_data_in_leaf':random.randint(20,60),  # minimal number of data in one leaf. Can be used to deal with over-fitting
#                 'min_split_gain': random.randint(0,1), # (float, optional (default=0.)) – Minimum loss reduction required to make a further partition on a leaf node of the tree.
#                 'metric': 'auc',
#                 'seed':7,
#                 'verbose': -1,          
#                 'boosting_type' : 'gbdt'
#             }

#         # train
#         gbm = lgb.train(params,
#                         lgb_train,
#                         num_boost_round=100000,
#                         valid_sets=lgb_eval,
#                         verbose_eval=False,
#                         early_stopping_rounds=100)

#         # predict (on test)
#         ypredlgbm = gbm.predict(xtest, num_iteration=gbm.best_iteration)
#         lgbauc = roc_auc_score(ytest, ypredlgbm)
#         if lgbauc>best_auc:
#             best_auc = lgbauc
#             best_params = params
#     except:
#         print("ex")
#         continue

# print("========================================")
# print(best_auc)
# print(best_params)

############################
# Final model 

params = {'bagging_fraction': 0.501449303, 
    'bagging_freq': 36, 
    'boosting_type': 'gbdt', 
    'feature_fraction': 0.386965536, 
    'lambda_l1': 0.7, 
    'learning_rate': 0.054908359, 
    'max_bin': 2, 
    'max_depth': 2, 
    'min_data_in_leaf': 1, 
    'min_sum_hessian_in_leaf': 0, 
    'num_leaves': 39, 
    'objective': 'regression', 
    'verbose': -1,
    'min_split_gain': 0,
    'metric': 'auc',
#    'verbose': -1,
    'seed':7,
 #   'boosting_type' : 'gbdt',
}
#               print("check5")
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100000,
                valid_sets=lgb_eval,#test set
                verbose_eval=False,# no commenting while computing
                early_stopping_rounds=100)

ypredlgbm = gbm.predict(xtest, num_iteration=gbm.best_iteration)
lgbauc = roc_auc_score(ytest, ypredlgbm)
lgbclasses = np.around(ypredlgbm, decimals=0)
#print(i)
        
print("========================================")
print("AUC LGB  : %.3f" %lgbauc)
AUC="AUC LGB  : %.3f" %lgbauc
AUC=lgbauc
print(AUC)
print("LIGHTGBM")
tnlgb, fplgb, fnlgb, tplgb = confusion_matrix(ytest, lgbclasses).ravel() #y_true, y_pred
print("True negative: %s, False positive: %s, False negative: %s, True positive %s | share false %.2f" %(tnlgb, fplgb, fnlgb, tplgb, ((fplgb+fnlgb)/(fplgb+fnlgb+tplgb+tnlgb))))

unique, counts = np.unique(lgbclasses, return_counts=True)
##    print("Tabel of classes from LGB (w count):")
print(np.asarray((unique, counts)).T)
print("Min. prob. : %.3f" %ypredlgbm.min())
print("Max. prob. : %.3f" %ypredlgbm.max())
#I.append(round(i,3))
#A.append(AUC)
#kay.append(round(k,3))
#    en.append(round(j,3))
#         en.append(round(n,2))
#            em.append(round(m,2))
#mini.append(round(ypredlgbm.min(),2))
#maxi.append(round(ypredlgbm.max(),2))
    
# FEATURE IMPORTANCE LIGHTGBM
# Look at important features in lightGBM

# FEATURE IMPORTANCE LIGHTGBM
# Look at important features in lightGBM
import matplotlib.pyplot as plt
lgb.plot_importance(gbm, max_num_features=55)
plt.show()






